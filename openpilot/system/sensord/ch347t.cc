#include <dlfcn.h>
#include <fcntl.h>
#include <glob.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "common/ratekeeper.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

ExitHandler do_exit;

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------
static constexpr uint8_t WHO_AM_I_REG = 0x0F;
static constexpr uint8_t LSM6_ADDRS[] = {0x6B, 0x6A};
static constexpr uint8_t WHO_AM_I_IDS[] = {0x69, 0x6A};
static constexpr int DEFAULT_I2C_BUS_IMU = 1;
static constexpr int MAX_STALE_FRAMES = 3;
static constexpr int MAX_CONSECUTIVE_ERRORS = 10;
static constexpr int MAX_REINIT_ATTEMPTS = 3;
static constexpr int CH347_WAIT_SECONDS = 30;
static constexpr int CH347_RECHECK_INTERVAL_MS = 5000;
static constexpr float ACCEL_SCALE = 9.81f * 2.0f / (1 << 15);
static constexpr float GYRO_SCALE = (8.75f / 1000.0f) * (M_PI / 180.0f);

// ---- Auto gyro zero-rate bias calibration (boot-time) ----
// Static detection: variance of gyro magnitude over the window must stay below
// this threshold (rad/s) for the samples to be accepted as "still".
static constexpr float CALIB_STATIC_GYRO_STD = 0.015f;   // 1-sigma rad/s
static constexpr int CALIB_SKIP_SECONDS = 1;             // skip first second after open
static constexpr int CALIB_COLLECT_SECONDS = 5;          // ~500 samples at 104 Hz
static constexpr int CALIB_SAMPLE_RATE_HZ = 104;
static constexpr int CALIB_MIN_SAMPLES = 200;            // require at least this many accepted


static inline int16_t parse_16bit(uint8_t lsb, uint8_t msb) {
  return static_cast<int16_t>(static_cast<uint16_t>(msb) << 8 | lsb);
}

// Cached sensor data for stale-frame fallback (scaled values)
struct SensorCache {
  float acc_v[3] = {};
  float gyro_v[3] = {};
  bool valid = false;
};

// -------------------------------------------------------------------
// Abstract LSM6DS3 interface
// -------------------------------------------------------------------
class LSM6DS3 {
public:
  virtual ~LSM6DS3() = default;
  virtual void open() = 0;
  virtual void init_sensor() = 0;
  virtual void shutdown() = 0;
  virtual uint8_t read_u8(uint8_t reg) = 0;
  virtual void write_u8(uint8_t reg, uint8_t val) = 0;
  virtual std::vector<uint8_t> read_block(uint8_t start_reg, int length) = 0;
  virtual cereal::SensorEventData::SensorSource source() const = 0;
};

// -------------------------------------------------------------------
// CH347 USB-I2C implementation
// -------------------------------------------------------------------
class CH347LSM6 : public LSM6DS3 {
public:
  CH347LSM6(const std::string &dev_path, void *lib_handle)
    : dev_path_(dev_path), lib_handle_(lib_handle), fd_(-1), addr_(0),
      source_(cereal::SensorEventData::SensorSource::LSM6DS3) {
    open_dev_ = (CH347OpenDevice_t)dlsym(lib_handle_, "CH347OpenDevice");
    close_dev_ = (CH347CloseDevice_t)dlsym(lib_handle_, "CH347CloseDevice");
    set_timeout_ = (CH34xSetTimeout_t)dlsym(lib_handle_, "CH34xSetTimeout");
    i2c_set_ = (CH347I2C_Set_t)dlsym(lib_handle_, "CH347I2C_Set");
    i2c_set_ignore_nack_ = (CH347I2C_SetIgnoreNack_t)dlsym(lib_handle_, "CH347I2C_SetIgnoreNack");
    i2c_set_stretch_ = (CH347I2C_SetStretch_t)dlsym(lib_handle_, "CH347I2C_SetStretch");
    stream_i2c_ = (CH347StreamI2C_t)dlsym(lib_handle_, "CH347StreamI2C");
    stream_i2c_ret_ack_ = (CH347StreamI2C_RetAck_t)dlsym(lib_handle_, "CH347StreamI2C_RetAck");
  }

  void open() override {
    fd_ = open_dev_(dev_path_.c_str());
    if (fd_ < 0) throw std::runtime_error("failed to open " + dev_path_);
    if (!set_timeout_(fd_, 2000, 2000)) throw std::runtime_error("CH34xSetTimeout failed");
    if (!i2c_set_(fd_, 0x01)) throw std::runtime_error("CH347I2C_Set failed");
    i2c_set_ignore_nack_(fd_, 1);
    i2c_set_stretch_(fd_, true);
    util::sleep_for(20);

    for (int attempt = 0; attempt < 8; attempt++) {
      for (uint8_t addr : LSM6_ADDRS) {
        try {
          uint8_t wb[] = {static_cast<uint8_t>(addr << 1), WHO_AM_I_REG};
          auto r = stream_read(wb, 2, 1);
          uint8_t who = r[0];
          for (uint8_t id : WHO_AM_I_IDS) {
            if (who == id) {
              addr_ = addr;
              source_ = (who == 0x6A)
                ? cereal::SensorEventData::SensorSource::LSM6DS3TRC
                : cereal::SensorEventData::SensorSource::LSM6DS3;
              LOG("CH347 LSM6DS3 detected at addr=0x%02X, who=0x%02X", addr, who);
              return;
            }
          }
        } catch (...) { continue; }
      }
      util::sleep_for(20);
    }
    throw std::runtime_error("CH347 LSM6DS3 not detected on 0x6A/0x6B");
  }

  void init_sensor() override {
    write_u8(0x12, 0x01);
    util::sleep_for(100);
    write_u8(0x12, 0x04);
    write_u8(0x10, 0x40);
    write_u8(0x11, 0x40);
  }

  void shutdown() override {
    if (fd_ >= 0) {
      try { write_u8(0x10, 0x00); } catch (...) {}
      try { write_u8(0x11, 0x00); } catch (...) {}
      close_dev_(fd_);
      fd_ = -1;
    }
  }

  uint8_t read_u8(uint8_t reg) override {
    uint8_t wb[] = {static_cast<uint8_t>(addr_ << 1), reg};
    auto r = stream_read(wb, 2, 1);
    return r[0];
  }

  void write_u8(uint8_t reg, uint8_t val) override {
    uint8_t wb[] = {static_cast<uint8_t>(addr_ << 1), reg, val};
    stream_write(wb, 3);
  }

  std::vector<uint8_t> read_block(uint8_t start_reg, int length) override {
    uint8_t addr_write[] = {static_cast<uint8_t>(addr_ << 1 | 0x00), start_reg};
    int ack = 0;
    if (!stream_i2c_ret_ack_(fd_, 2, addr_write, 0, nullptr, &ack))
      throw std::runtime_error("CH347 I2C block write address failed");
    uint8_t addr_read[] = {static_cast<uint8_t>(addr_ << 1 | 0x01)};
    std::vector<uint8_t> rbuf(length);
    if (!stream_i2c_ret_ack_(fd_, 1, addr_read, length, rbuf.data(), &ack))
      throw std::runtime_error("CH347 I2C block read data failed");
    return rbuf;
  }

  cereal::SensorEventData::SensorSource source() const override { return source_; }

private:
  std::vector<uint8_t> stream_read(const uint8_t *write_bytes, int write_len, int read_len) {
    std::vector<uint8_t> rbuf(read_len);
    int ack = 0;
    if (!stream_i2c_ret_ack_(fd_, write_len, const_cast<uint8_t *>(write_bytes),
                             read_len, rbuf.data(), &ack))
      throw std::runtime_error("CH347 I2C read failed");
    return rbuf;
  }

  void stream_write(const uint8_t *write_bytes, int write_len) {
    if (!stream_i2c_(fd_, write_len, const_cast<uint8_t *>(write_bytes), 0, nullptr))
      throw std::runtime_error("CH347 I2C write failed");
  }

  // CH347 function pointer types
  typedef int (*CH347OpenDevice_t)(const char *);
  typedef bool (*CH347CloseDevice_t)(int);
  typedef bool (*CH34xSetTimeout_t)(int, uint32_t, uint32_t);
  typedef bool (*CH347I2C_Set_t)(int, int);
  typedef bool (*CH347I2C_SetIgnoreNack_t)(int, uint8_t);
  typedef bool (*CH347I2C_SetStretch_t)(int, bool);
  typedef bool (*CH347StreamI2C_t)(int, int, void *, int, void *);
  typedef bool (*CH347StreamI2C_RetAck_t)(int, int, void *, int, void *, int *);

  std::string dev_path_;
  void *lib_handle_;
  int fd_;
  uint8_t addr_;
  cereal::SensorEventData::SensorSource source_;
  CH347OpenDevice_t open_dev_ = nullptr;
  CH347CloseDevice_t close_dev_ = nullptr;
  CH34xSetTimeout_t set_timeout_ = nullptr;
  CH347I2C_Set_t i2c_set_ = nullptr;
  CH347I2C_SetIgnoreNack_t i2c_set_ignore_nack_ = nullptr;
  CH347I2C_SetStretch_t i2c_set_stretch_ = nullptr;
  CH347StreamI2C_t stream_i2c_ = nullptr;
  CH347StreamI2C_RetAck_t stream_i2c_ret_ack_ = nullptr;
};

// -------------------------------------------------------------------
// Direct Linux I2C implementation (fallback via /dev/i2c-N)
// -------------------------------------------------------------------
class I2CLSM6 : public LSM6DS3 {
public:
  I2CLSM6(int bus, uint8_t addr)
    : bus_(bus), addr_(addr), fd_(-1),
      source_(addr == 0x6A
        ? cereal::SensorEventData::SensorSource::LSM6DS3TRC
        : cereal::SensorEventData::SensorSource::LSM6DS3) {}

  void open() override {
    char path[32];
    snprintf(path, sizeof(path), "/dev/i2c-%d", bus_);
    fd_ = ::open(path, O_RDWR);
    if (fd_ < 0) throw std::runtime_error(std::string("failed to open ") + path);
    if (ioctl(fd_, I2C_SLAVE, addr_) < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("I2C_SLAVE failed");
    }
  }

  void init_sensor() override {
    write_u8(0x12, 0x01);
    util::sleep_for(100);
    write_u8(0x12, 0x04);
    write_u8(0x10, 0x40);
    write_u8(0x11, 0x40);
  }

  void shutdown() override {
    if (fd_ >= 0) {
      try { write_u8(0x10, 0x00); } catch (...) {}
      try { write_u8(0x11, 0x00); } catch (...) {}
      ::close(fd_);
      fd_ = -1;
    }
  }

  uint8_t read_u8(uint8_t reg) override {
    if (::write(fd_, &reg, 1) != 1)
      throw std::runtime_error("I2C write reg failed");
    uint8_t val;
    if (::read(fd_, &val, 1) != 1)
      throw std::runtime_error("I2C read failed");
    return val;
  }

  void write_u8(uint8_t reg, uint8_t val) override {
    uint8_t buf[] = {reg, val};
    if (::write(fd_, buf, 2) != 2)
      throw std::runtime_error("I2C write failed");
  }

  std::vector<uint8_t> read_block(uint8_t start_reg, int length) override {
    struct i2c_msg msgs[2];
    struct i2c_rdwr_ioctl_data msgset;
    msgs[0].addr = addr_;
    msgs[0].flags = 0;
    msgs[0].len = 1;
    msgs[0].buf = &start_reg;
    std::vector<uint8_t> rbuf(length);
    msgs[1].addr = addr_;
    msgs[1].flags = I2C_M_RD;
    msgs[1].len = length;
    msgs[1].buf = rbuf.data();
    msgset.msgs = msgs;
    msgset.nmsgs = 2;
    if (ioctl(fd_, I2C_RDWR, &msgset) < 0)
      throw std::runtime_error("I2C_RDWR failed");
    return rbuf;
  }

  cereal::SensorEventData::SensorSource source() const override { return source_; }

private:
  int bus_;
  uint8_t addr_;
  int fd_;
  cereal::SensorEventData::SensorSource source_;
};

// -------------------------------------------------------------------
// CH347 library path resolution
// -------------------------------------------------------------------
static std::string get_ch347_lib_path() {
  char exe_path[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  std::string project_root;
  if (len > 0) {
    exe_path[len] = '\0';
    std::string exe(exe_path);
    auto pos = exe.rfind('/');
    if (pos != std::string::npos) {
      std::string dir = exe.substr(0, pos);
      pos = dir.rfind('/');
      if (pos != std::string::npos) {
        dir = dir.substr(0, pos);
        pos = dir.rfind('/');
        if (pos != std::string::npos)
          project_root = dir.substr(0, pos);
      }
    }
  }
  if (project_root.empty()) project_root = ".";

  std::string arch_dir;
  struct utsname uts;
  if (uname(&uts) == 0) {
    std::string machine(uts.machine);
    if (machine == "x86_64" || machine == "amd64")
      arch_dir = "x64";
    else if (machine.find("arm") == 0)
      arch_dir = (machine.find("v7l") != std::string::npos || machine.find("hf") != std::string::npos)
                 ? "arm-gnueabihf" : "arm-gnueabi";
    else if (machine.find("aarch64") == 0)
      arch_dir = "aarch64";
    else
      arch_dir = "x86";
  } else {
    arch_dir = "x64";
  }

  std::string lib_path = project_root + "/third_party/ch347/lib/" + arch_dir + "/dynamic/libch347.so";
  if (!util::file_exists(lib_path))
    lib_path = project_root + "/third_party/ch347/lib/x64/dynamic/libch347.so";
  return lib_path;
}

// -------------------------------------------------------------------
// CH347 backend detection
// -------------------------------------------------------------------
static bool detect_ch347_backend(std::string &dev_path, std::string &lib_path) {
  const char *env_dev = std::getenv("SENSORD_CH347_DEV");
  if (env_dev && env_dev[0] != '\0') {
    dev_path = env_dev;
  } else {
    glob_t globbuf;
    bool found = false;
    if (glob("/dev/ch34x_pis*", 0, nullptr, &globbuf) == 0 && globbuf.gl_pathc > 0) {
      dev_path = globbuf.gl_pathv[0];
      found = true;
    }
    globfree(&globbuf);
    if (!found) {
      // Also try /dev/ttyACM* (some CH347 appear as ACM)
      if (glob("/dev/ttyACM*", 0, nullptr, &globbuf) == 0 && globbuf.gl_pathc > 0) {
        dev_path = globbuf.gl_pathv[0];
        found = true;
      }
      globfree(&globbuf);
    }
    if (!found) return false;
  }

  const char *env_lib = std::getenv("SENSORD_CH347_LIB");
  if (env_lib && env_lib[0] != '\0') {
    lib_path = env_lib;
  } else {
    lib_path = get_ch347_lib_path();
  }

  if (!util::file_exists(dev_path)) { LOGE("CH347 dev does not exist: %s", dev_path.c_str()); return false; }
  if (!util::file_exists(lib_path)) { LOGE("CH347 lib does not exist: %s", lib_path.c_str()); return false; }
  return true;
}

// -------------------------------------------------------------------
// Direct I2C bus detection
// -------------------------------------------------------------------
static int detect_imu_i2c_bus(uint8_t &addr) {
  const char *env_bus = std::getenv("SENSORD_I2C_BUS");
  if (env_bus) {
    int bus = atoi(env_bus);
    LOG("Probing IMU on env bus %d", bus);
    for (uint8_t a : LSM6_ADDRS) {
      char path[32];
      snprintf(path, sizeof(path), "/dev/i2c-%d", bus);
      int fd = ::open(path, O_RDWR);
      if (fd < 0) continue;
      if (ioctl(fd, I2C_SLAVE, a) < 0) { ::close(fd); continue; }
      uint8_t reg = WHO_AM_I_REG;
      ::write(fd, &reg, 1);
      uint8_t who;
      if (::read(fd, &who, 1) == 1) {
        ::close(fd);
        for (uint8_t id : WHO_AM_I_IDS) {
          if (who == id) { addr = a; LOG("LSM6DS3 on /dev/i2c-%d, addr=0x%02X", bus, a); return bus; }
        }
      } else {
        ::close(fd);
      }
    }
    LOGW("LSM6DS3 not found on bus %d", bus);
  }

  // Scan all /dev/i2c-*
  glob_t globbuf;
  std::vector<int> buses;
  if (glob("/dev/i2c-*", 0, nullptr, &globbuf) == 0) {
    for (size_t i = 0; i < globbuf.gl_pathc; i++) {
      std::string p = globbuf.gl_pathv[i];
      auto pos = p.rfind('-');
      if (pos != std::string::npos) {
        try { buses.push_back(std::stoi(p.substr(pos + 1))); } catch (...) {}
      }
    }
    globfree(&globbuf);
  }

  // Prioritize DEFAULT bus, then probe all
  std::vector<int> ordered;
  for (int b : buses) { if (b == DEFAULT_I2C_BUS_IMU) { ordered.push_back(b); break; } }
  for (int b : buses) { if (b != DEFAULT_I2C_BUS_IMU) ordered.push_back(b); }
  if (ordered.empty()) ordered.push_back(DEFAULT_I2C_BUS_IMU);

  for (int bus : ordered) {
    for (uint8_t a : LSM6_ADDRS) {
      char path[32];
      snprintf(path, sizeof(path), "/dev/i2c-%d", bus);
      int fd = ::open(path, O_RDWR);
      if (fd < 0) continue;
      if (ioctl(fd, I2C_SLAVE, a) < 0) { ::close(fd); continue; }
      uint8_t reg = WHO_AM_I_REG;
      ::write(fd, &reg, 1);
      uint8_t who;
      if (::read(fd, &who, 1) == 1) {
        ::close(fd);
        for (uint8_t id : WHO_AM_I_IDS) {
          if (who == id) { addr = a; LOG("LSM6DS3 on /dev/i2c-%d, addr=0x%02X", bus, a); return bus; }
        }
      } else {
        ::close(fd);
      }
    }
  }

  LOGW("LSM6DS3 not found, defaulting to bus %d addr 0x%02X", DEFAULT_I2C_BUS_IMU, LSM6_ADDRS[0]);
  addr = LSM6_ADDRS[0];
  return DEFAULT_I2C_BUS_IMU;
}

// -------------------------------------------------------------------
// Publish helpers (use scaled values directly)
// -------------------------------------------------------------------
static void publish_accelerometer(PubMaster &pm,
                                  cereal::SensorEventData::SensorSource source,
                                  float v0, float v1, float v2) {
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto se = evt.initAccelerometer();
  se.setVersion(1);
  se.setSensor(1);
  se.setType(1);
  se.setSource(source);
  se.setTimestamp(evt.getLogMonoTime());
  auto acc = se.initAcceleration();
  auto v = acc.initV(3);
  v.set(0, v0);
  v.set(1, v1);
  v.set(2, v2);
  acc.setStatus(1);
  pm.send("accelerometer", msg);
}

static void publish_gyroscope(PubMaster &pm,
                               cereal::SensorEventData::SensorSource source,
                               float v0, float v1, float v2) {
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto se = evt.initGyroscope();
  se.setVersion(2);
  se.setSensor(5);
  se.setType(16);
  se.setSource(source);
  se.setTimestamp(evt.getLogMonoTime());
  auto gyro = se.initGyroUncalibrated();
  auto v = gyro.initV(3);
  v.set(0, v0);
  v.set(1, v1);
  v.set(2, v2);
  gyro.setStatus(1);
  pm.send("gyroscope", msg);
}

static void publish_temperature(PubMaster &pm,
                                cereal::SensorEventData::SensorSource source,
                                float raw_t) {
  float tscale = (source == cereal::SensorEventData::SensorSource::LSM6DS3) ? 16.0f : 256.0f;
  MessageBuilder msg;
  auto evt = msg.initEvent();
  auto se = evt.initTemperatureSensor();
  se.setVersion(1);
  se.setSource(source);
  se.setTimestamp(evt.getLogMonoTime());
  se.setTemperature(25.0f + raw_t / tscale);
  pm.send("temperatureSensor", msg);
}

// -------------------------------------------------------------------
// Boot-time gyro zero-rate bias auto-calibration
// -------------------------------------------------------------------
static std::string calib_json_path() {
  const char *env = std::getenv("IMU_CALIB_JSON");
  if (env && env[0] != '\0') return env;
  char pwd[PATH_MAX];
  return (getcwd(pwd, sizeof(pwd)) != nullptr)
    ? std::string(pwd) + "/imu_calibration.json" : "imu_calibration.json";
}

static std::pair<bool, float> find_float(const std::string &json, const std::string &key) {
  std::string pat1 = "\"" + key + "\":";
  auto i = json.find(pat1);
  if (i == std::string::npos) return {false, 0.f};
  i += pat1.size();
  auto j = json.find_first_of(",}]", i);
  if (j == std::string::npos) return {false, 0.f};
  try { return {true, std::stof(json.substr(i, j - i))}; } catch (...) { return {false, 0.f}; }
}

static void load_gyro_bias(float bias[3]) {
  bias[0] = bias[1] = bias[2] = 0.f;
  std::ifstream f(calib_json_path());
  if (!f) return;
  std::stringstream ss; ss << f.rdbuf(); std::string json = ss.str();
  // imuBiasGyro array: [x, y, z]
  std::string pat = "\"imuBiasGyro\":[";
  auto i = json.find(pat);
  if (i == std::string::npos) return;
  i += pat.size();
  std::vector<float> vals;
  while (vals.size() < 3 && i < json.size()) {
    auto j = json.find_first_of(",]", i);
    if (j == std::string::npos) break;
    try { vals.push_back(std::stof(json.substr(i, j - i))); } catch (...) { vals.push_back(0.f); }
    i = j + 1;
  }
  for (size_t k = 0; k < vals.size() && k < 3; k++) bias[k] = vals[k];
}

static void save_gyro_bias(const float bias[3]) {
  std::string json = "{\n"
    "  \"imuCalibMatrix\": [1,0,0, 0,1,0, 0,0,1],\n"
    "  \"imuBiasAccel\": [0.0, 0.0, 0.0],\n"
    "  \"imuBiasGyro\": [" +
    std::to_string((double)bias[0]) + ", " + std::to_string((double)bias[1]) + ", " +
    std::to_string((double)bias[2]) + "]\n}\n";
  std::ofstream f(calib_json_path());
  if (!f) { LOGW("IMU: could not write %s", calib_json_path().c_str()); return; }
  f << json;
  LOG("IMU: saved gyro bias (%s)", calib_json_path().c_str());
}

// Called right after open()+init_sensor(). Collects ~5s of data; if the device
// is stationary (gyro magnitude 1-sigma below threshold), updates the gyro
// zero-rate bias in imu_calibration.json. If moving, keeps the existing bias.
static void auto_calibrate_gyro_bias(LSM6DS3 &sensor, const std::string &backend) {
  float existing[3];
  load_gyro_bias(existing);
  LOG("IMU: boot auto-calibrate (%s), existing bias=[%+.4f %+.4f %+.4f]",
      backend.c_str(), existing[0], existing[1], existing[2]);

  // Skip the first second for sensor stabilization.
  util::sleep_for(CALIB_SKIP_SECONDS * 1000);

  std::vector<float> gx, gy, gz;
  int samples = CALIB_COLLECT_SECONDS * CALIB_SAMPLE_RATE_HZ;
  for (int i = 0; i < samples && !do_exit; i++) {
    bool ok = false;
    try {
      auto b = sensor.read_block(0x22, 6);
      float x = parse_16bit(b[0], b[1]);
      float y = parse_16bit(b[2], b[3]);
      float z = parse_16bit(b[4], b[5]);
      gx.push_back(y * GYRO_SCALE);
      gy.push_back(-x * GYRO_SCALE);
      gz.push_back(z * GYRO_SCALE);
      ok = true;
    } catch (...) {}
    if (!ok) util::sleep_for(10);
    else util::sleep_for(1000 / CALIB_SAMPLE_RATE_HZ);
  }
  int n = (int)gx.size();
  if (n < CALIB_MIN_SAMPLES) {
    LOGW("IMU: insufficient samples (%d < %d), keeping existing bias", n, CALIB_MIN_SAMPLES);
    return;
  }
  auto mean = [&](const std::vector<float> &v) { float s = 0; for (float x : v) s += x; return s / v.size(); };
  auto sd = [&](const std::vector<float> &v, float m) {
    float s = 0; for (float x : v) s += (x - m) * (x - m); return std::sqrt(s / v.size());
  };
  float mx = mean(gx), my = mean(gy), mz = mean(gz);
  float sx = sd(gx, mx), sy = sd(gy, my), sz = sd(gz, mz);
  // Combined 1-sigma of gyro magnitude vector ~ sqrt(mean of per-axis variance).
  float mag_std = std::sqrt((sx * sx + sy * sy + sz * sz) / 3.0f);
  LOG("IMU: samples=%d stds=[%+.5f %+.5f %+.5f] mag_std=%+.5f", n, sx, sy, sz, mag_std);
  if (mag_std >= CALIB_STATIC_GYRO_STD) {
    LOGW("IMU: not stationary (mag_std=%.5f >= 0.015), keeping existing bias [%+.4f %+.4f %+.4f]",
         mag_std, existing[0], existing[1], existing[2]);
    return;
  }
  float new_bias[3] = {mx, my, mz};
  LOG("IMU: stationary, bias %+.4f -> %+.4f %+.4f %+.4f", existing[0], new_bias[0], new_bias[1], new_bias[2]);
  save_gyro_bias(new_bias);
}

// -------------------------------------------------------------------
// Reinitialize sensor with retry
// -------------------------------------------------------------------
static bool reinit_sensor(LSM6DS3 &sensor, const std::string &name) {
  try {
    LOGW("%s: reinitializing sensor...", name.c_str());
    sensor.shutdown();
    util::sleep_for(500);
    sensor.open();
    sensor.init_sensor();
    LOG("%s: sensor reinitialized successfully", name.c_str());
    return true;
  } catch (const std::exception &e) {
    LOGE("%s: reinitialization failed: %s", name.c_str(), e.what());
    return false;
  }
}

// -------------------------------------------------------------------
// Wait for CH347 device (non-blocking if do_exit is set)
// -------------------------------------------------------------------
static bool wait_for_ch347(std::string &dev_path, std::string &lib_path) {
  LOG("Waiting up to %ds for CH347 device...", CH347_WAIT_SECONDS);
  for (int i = 0; i < CH347_WAIT_SECONDS && !do_exit; i++) {
    if (detect_ch347_backend(dev_path, lib_path)) {
      LOG("CH347 device detected after %ds", i);
      return true;
    }
    util::sleep_for(1000);
  }
  return false;
}

// -------------------------------------------------------------------
// Main sensor read loop (shared by both backends)
// Returns true on clean exit, false on fatal error
// -------------------------------------------------------------------
static bool sensor_read_loop(LSM6DS3 &sensor, PubMaster &pm, RateKeeper &rk,
                             const std::string &backend_name,
                             std::atomic<bool> &switch_backend) {
  int temp_ticks = 0;
  int consecutive_errors = 0;
  int reinit_attempts = 0;
  SensorCache cache;
  float gyro_bias[3] = {0.f, 0.f, 0.f};

  try {
    sensor.open();
    sensor.init_sensor();
    LOG("Using %s backend", backend_name.c_str());
  } catch (const std::exception &e) {
    LOGE("%s: open/init failed: %s", backend_name.c_str(), e.what());
    return false;
  }

  // Boot-time auto zero-bias calibration (only once when the backend starts).
  auto_calibrate_gyro_bias(sensor, backend_name);
  load_gyro_bias(gyro_bias);
  if (gyro_bias[0] != 0.f || gyro_bias[1] != 0.f || gyro_bias[2] != 0.f)
    LOG("IMU: applying gyro bias [-%+.4f -%+.4f -%+.4f]", gyro_bias[0], gyro_bias[1], gyro_bias[2]);

  while (!do_exit && !switch_backend) {
    bool i2c_error = false;
    bool fresh_acc = false, fresh_gyro = false;

    uint8_t status = 0;
    try { status = sensor.read_u8(0x1E); }
    catch (...) { i2c_error = true; }

    if ((status & 0x01) && !i2c_error) {
      try {
        auto b = sensor.read_block(0x28, 6);
        float x = parse_16bit(b[0], b[1]);
        float y = parse_16bit(b[2], b[3]);
        float z = parse_16bit(b[4], b[5]);
        // Axis mapping: [y, -x, z]
        cache.acc_v[0] = y * ACCEL_SCALE;
        cache.acc_v[1] = -x * ACCEL_SCALE;
        cache.acc_v[2] = z * ACCEL_SCALE;
        cache.valid = true;
        fresh_acc = true;
      } catch (...) { i2c_error = true; }
    }

    if ((status & 0x02) && !i2c_error) {
      try {
        auto b = sensor.read_block(0x22, 6);
        float x = parse_16bit(b[0], b[1]);
        float y = parse_16bit(b[2], b[3]);
        float z = parse_16bit(b[4], b[5]);
        cache.gyro_v[0] = y * GYRO_SCALE - gyro_bias[0];
        cache.gyro_v[1] = -x * GYRO_SCALE - gyro_bias[1];
        cache.gyro_v[2] = z * GYRO_SCALE - gyro_bias[2];
        fresh_gyro = true;
      } catch (...) { i2c_error = true; }
    }

    if (i2c_error) {
      consecutive_errors++;
    } else {
      consecutive_errors = 0;
      reinit_attempts = 0;
    }

    bool send_stale = i2c_error && consecutive_errors <= MAX_STALE_FRAMES;

    if (fresh_acc) {
      publish_accelerometer(pm, sensor.source(), cache.acc_v[0], cache.acc_v[1], cache.acc_v[2]);
    } else if (send_stale && cache.valid) {
      publish_accelerometer(pm, sensor.source(), cache.acc_v[0], cache.acc_v[1], cache.acc_v[2]);
    }

    if (fresh_gyro) {
      publish_gyroscope(pm, sensor.source(), cache.gyro_v[0], cache.gyro_v[1], cache.gyro_v[2]);
    } else if (send_stale && cache.valid) {
      publish_gyroscope(pm, sensor.source(), cache.gyro_v[0], cache.gyro_v[1], cache.gyro_v[2]);
    }

    if (send_stale) {
      LOGW("%s: I2C error, sending stale data (%d/%d)",
           backend_name.c_str(), consecutive_errors, MAX_STALE_FRAMES);
    }

    // Temperature (every 10 ticks)
    temp_ticks++;
    if (temp_ticks >= 10) {
      temp_ticks = 0;
      try {
        auto tb = sensor.read_block(0x20, 2);
        float raw_t = parse_16bit(tb[0], tb[1]);
        publish_temperature(pm, sensor.source(), raw_t);
      } catch (...) {}
    }

    // Periodic debug log to stderr (every 1 second)
    {
      static uint64_t last_log_t = 0;
      uint64_t now = nanos_since_boot();
      if (now - last_log_t > 1e9) {
        last_log_t = now;
        if (cache.valid) {
          fprintf(stderr, "[%s] ACC: [%7.2f, %7.2f, %7.2f] m/s²  GYRO: [%7.3f, %7.3f, %7.3f] rad/s\n",
              backend_name.c_str(),
              cache.acc_v[0], cache.acc_v[1], cache.acc_v[2],
              cache.gyro_v[0], cache.gyro_v[1], cache.gyro_v[2]);
        } else {
          fprintf(stderr, "[%s] Waiting for sensor data...\n", backend_name.c_str());
        }
        fflush(stderr);
      }
    }

    // Auto-reinit on persistent errors
    if (consecutive_errors > MAX_CONSECUTIVE_ERRORS) {
      LOGE("%s: %d consecutive errors, reinit (%d/%d)",
           backend_name.c_str(), consecutive_errors,
           reinit_attempts + 1, MAX_REINIT_ATTEMPTS);
      if (reinit_attempts < MAX_REINIT_ATTEMPTS) {
        if (reinit_sensor(sensor, backend_name)) {
          consecutive_errors = 0;
          reinit_attempts = 0;
          continue;
        }
        reinit_attempts++;
      } else {
        LOGE("%s: reinit failed after %d attempts, giving up",
             backend_name.c_str(), MAX_REINIT_ATTEMPTS);
        return false;
      }
    }

    rk.keepTime();
  }
  return true;
}

// -------------------------------------------------------------------
// Entry point
// -------------------------------------------------------------------
int main(int argc, char **argv) {
  setpriority(PRIO_PROCESS, 0, -15);
  PubMaster publisher({"accelerometer", "gyroscope", "temperatureSensor"});
  std::atomic<bool> switch_backend{false};

  while (!do_exit) {
    // Reduce priority after watchdog restart
    setpriority(PRIO_PROCESS, 0, -15);

    // ---- Phase 1: Try CH347 ----
    std::string dev_path, lib_path;
    if (wait_for_ch347(dev_path, lib_path)) {
      void *lib_handle = dlopen(lib_path.c_str(), RTLD_NOW);
      if (lib_handle) {
        LOG("CH347 backend available, starting...");
        CH347LSM6 sensor(dev_path, lib_handle);
        RateKeeper rk("ch347d", 104.0f);
        bool ok = sensor_read_loop(sensor, publisher, rk, "CH347", switch_backend);
        sensor.shutdown();
        dlclose(lib_handle);
        if (ok) { LOG("CH347: clean exit"); break; }
        LOGW("CH347 backend failed, falling back to direct I2C");
      } else {
        LOGE("Failed to load CH347 library: %s", dlerror());
      }
    } else {
      LOGW("CH347 not available after waiting");
    }

    if (do_exit) break;

    // ---- Phase 2: Fallback to direct I2C with auto-upgrade ----
    uint8_t i2c_addr = LSM6_ADDRS[0];
    int i2c_bus = detect_imu_i2c_bus(i2c_addr);
    LOG("Starting direct I2C backend (bus=%d, addr=0x%02X)", i2c_bus, i2c_addr);

    // Background thread: periodically check if CH347 becomes available
    std::atomic<bool> ch347_found{false};
    std::thread ch347_checker([&]() {
      while (!do_exit && !ch347_found) {
        std::string ch_dev, ch_lib;
        if (detect_ch347_backend(ch_dev, ch_lib)) {
          LOG("CH347 detected during I2C fallback, preparing switch...");
          ch347_found = true;
          switch_backend = true;
          break;
        }
        // Check every 5 seconds
        for (int i = 0; i < CH347_RECHECK_INTERVAL_MS / 100 && !do_exit && !ch347_found; i++)
          util::sleep_for(100);
      }
    });

    I2CLSM6 sensor(i2c_bus, i2c_addr);
    RateKeeper rk("ch347d", 104.0f);
    bool ok = sensor_read_loop(sensor, publisher, rk, "I2C", switch_backend);
    sensor.shutdown();

    ch347_found = true;  // signal checker to stop
    if (ch347_checker.joinable()) ch347_checker.join();

    if (ok) { LOG("I2C: clean exit"); break; }

    // If switch_backend was set (CH347 detected), restart outer loop to use CH347
    if (switch_backend) {
      LOGW("Switching from I2C to CH347...");
      switch_backend = false;
      continue;
    }

    // I2C failed fatally, wait before retry
    LOGW("I2C backend failed, retrying in 2s...");
    util::sleep_for(2000);
    continue;
  }

  LOG("sensord_ch347 exiting");
  return 0;
}
