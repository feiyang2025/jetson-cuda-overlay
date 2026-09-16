#include <dlfcn.h>
#include <fcntl.h>
#include <glob.h>
#include <linux/hidraw.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "common/ratekeeper.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "third_party/json11/json11.hpp"

ExitHandler do_exit;

// -------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------
static constexpr uint8_t WHO_AM_I_REG = 0x0F;
static constexpr uint8_t LSM6_ADDRS[] = {0x6B, 0x6A};
static constexpr uint8_t WHO_AM_I_IDS[] = {0x69, 0x6A};
static constexpr int MAX_STALE_FRAMES = 3;
static constexpr int MAX_CONSECUTIVE_ERRORS = 10;
static constexpr int MAX_REINIT_ATTEMPTS = 3;
static constexpr int CH347_WAIT_SECONDS = 30;
static constexpr int CH347_RECHECK_INTERVAL_MS = 5000;
// ---------------------------------------------------------------------------
// AGX Orin 本地 I2C 总线安全限制（2026-08-26 安全修复）
// 事故：旧逻辑 glob 扫描全部 /dev/i2c-* 并裸写探测，戳中设备树里没有任何子设备的
// 空总线 3190000(i2c-3)，触发 tegra-i2c transfer timed out + tegra-bpmp-i2c 失败，
// 随后整机硬件级断电（无 panic/OOM/关机日志）。详见 openpilot/IMU/ch347t_安全修复说明.md
// 实测补充：本机所有未接设备的外部总线（含 i2c-5/31b0000、i2c-6/31c0000）裸写
// 同样会触发超时风暴 —— 因此【默认完全不探测】本地 I2C；
// 仅当操作者显式设置 SENSORD_I2C_BUS 时才探测该单条总线，
// 且系统关键总线永远拒绝：
//   0(3160000) 1(c240000,HDMI/DDC) 3(3190000,事故总线) 4(BPMP 电源管理)
//   9~12(i2c-2-mux 复用通道) 13(SOC adapter)
static constexpr int FORBIDDEN_I2C_BUSES[] = {0, 1, 3, 4, 9, 10, 11, 12, 13};
static constexpr float ACCEL_SCALE = 9.81f * 2.0f / (1 << 15);
static constexpr float GYRO_SCALE = (8.75f / 1000.0f) * (M_PI / 180.0f);

// -------------------------------------------------------------------
// IMU 校准参数（可选，imu_calibration.json）
// 生成工具: tools/imu_calib/imu_calibration.py（多静止姿态椭球拟合）
// 应用: gyro_rad = (raw_dps - imuBiasGyro) * pi/180
//       accel_ms2 = imuCalibMatrix @ (raw_mps2 - 9.81*imuBiasAccel)
// 无此文件/解析失败时发原始值，行为与旧版完全一致。
// -------------------------------------------------------------------
struct ImuCalib {
  bool valid = false;
  float gyro_bias_dps[3] = {0.f, 0.f, 0.f};  // imuBiasGyro, °/s
  float accel_bias_g[3] = {0.f, 0.f, 0.f};   // imuBiasAccel, g
  float matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // imuCalibMatrix 3x3 行主序
};
static ImuCalib g_imu_calib;
// 运行时陀螺零偏(rad/s)：初始=json imuBiasGyro(或 0)，启动自动零偏校准完成后覆盖。
// 理由：LSM6DS3 零偏随温度漂移明显，一次 json 校准会"过期"，每次开机静置几秒自测更准。
static float g_runtime_gyro_bias_rad[3] = {0.f, 0.f, 0.f};

static std::string get_project_root() {
  char exe_path[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len <= 0) return ".";
  exe_path[len] = '\0';
  std::string exe(exe_path);
  for (int i = 0; i < 3; i++) {
    auto pos = exe.rfind('/');
    if (pos == std::string::npos) return ".";
    exe = exe.substr(0, pos);
  }
  return exe.empty() ? "." : exe;
}

static void load_imu_calibration(const std::string &path) {
  std::string content = util::read_file(path);
  if (content.empty()) {
    LOGW("imu_calibration.json not found (%s), using raw sensor values", path.c_str());
    return;
  }
  std::string err;
  json11::Json json = json11::Json::parse(content, err);
  if (!err.empty() || !json.is_object()) {
    LOGE("imu_calibration.json parse failed: %s", err.c_str());
    return;
  }
  auto gb = json["imuBiasGyro"];
  auto ab = json["imuBiasAccel"];
  auto m = json["imuCalibMatrix"];
  if (!gb.is_array() || gb.array_items().size() != 3 ||
      !ab.is_array() || ab.array_items().size() != 3 ||
      !m.is_array() || m.array_items().size() != 9) {
    LOGE("imu_calibration.json bad schema (need imuBiasGyro[3], imuBiasAccel[3], imuCalibMatrix[9])");
    return;
  }
  for (int i = 0; i < 3; i++) {
    g_imu_calib.gyro_bias_dps[i] = (float)gb.array_items()[i].number_value();
    g_imu_calib.accel_bias_g[i] = (float)ab.array_items()[i].number_value();
  }
  for (int i = 0; i < 9; i++)
    g_imu_calib.matrix[i] = (float)m.array_items()[i].number_value();
  g_imu_calib.valid = true;
  // json 零偏仅作启动初值，之后会被"启动自动零偏校准"覆盖（零偏随温度漂移）
  for (int i = 0; i < 3; i++)
    g_runtime_gyro_bias_rad[i] = g_imu_calib.gyro_bias_dps[i] * (float)(M_PI / 180.0);
  LOG("IMU calibration loaded: gyro_bias_dps=[%.4f,%.4f,%.4f] accel_bias_g=[%.4f,%.4f,%.4f]",
      g_imu_calib.gyro_bias_dps[0], g_imu_calib.gyro_bias_dps[1], g_imu_calib.gyro_bias_dps[2],
      g_imu_calib.accel_bias_g[0], g_imu_calib.accel_bias_g[1], g_imu_calib.accel_bias_g[2]);
}

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
    // 不同批次厂商库符号名不同：老库导出 CH34xSetTimeout，新库（IMU 部署包 2026-08）
    // 导出 CH347SetTimeout，两个都试
    set_timeout_ = (CH34xSetTimeout_t)dlsym(lib_handle_, "CH34xSetTimeout");
    if (!set_timeout_) set_timeout_ = (CH34xSetTimeout_t)dlsym(lib_handle_, "CH347SetTimeout");
    i2c_set_ = (CH347I2C_Set_t)dlsym(lib_handle_, "CH347I2C_Set");
    stream_i2c_ = (CH347StreamI2C_t)dlsym(lib_handle_, "CH347StreamI2C");
    // Optional symbols: absent in older libch347 builds (e.g. the aarch64 lib).
    // Fall back to plain CH347StreamI2C when missing instead of calling NULL.
    void *sym = dlsym(lib_handle_, "CH347StreamI2C_RetAck");
    if (sym) stream_i2c_ret_ack_ = (CH347StreamI2C_RetAck_t)sym;
    sym = dlsym(lib_handle_, "CH347I2C_SetIgnoreNack");
    if (sym) i2c_set_ignore_nack_ = (CH347I2C_SetIgnoreNack_t)sym;
    sym = dlsym(lib_handle_, "CH347I2C_SetStretch");
    if (sym) i2c_set_stretch_ = (CH347I2C_SetStretch_t)sym;

    if (!open_dev_ || !close_dev_ || !set_timeout_ || !i2c_set_ || !stream_i2c_)
      throw std::runtime_error("libch347 missing required symbols");
  }

  void open() override {
    // ---- 兼容两种厂商库调用约定 ----
    // 新库（IMU 部署包 2026-08）：CH347OpenDevice(ULONG idx)，内部拼 "/dev/hidraw<idx+2>"
    //   （参考 IMU/scripts/imu_lsm6ds3.py: idx = hidrawN - 2）
    // 旧库：CH347OpenDevice(const char *path)，直接传完整路径
    // 用 WHO_AM_I 实测成功与否决定采用哪种约定
    int hidraw_n = -1;
    std::sscanf(dev_path_.c_str(), "/dev/hidraw%d", &hidraw_n);
    if (hidraw_n >= 0) {
      typedef int (*OpenIdxT)(unsigned long);
      fd_ = ((OpenIdxT)(void *)open_dev_)((unsigned long)(long)(hidraw_n - 2));
      LOG("CH347 open(index convention): fd=%d", fd_);
      if (fd_ >= 0 && setup_and_detect()) return;
      if (fd_ >= 0) { close_dev_(fd_); fd_ = -1; }
    }
    // ---- 回退旧库约定（路径）----
    fd_ = open_dev_(dev_path_.c_str());
    if (fd_ < 0) throw std::runtime_error("failed to open " + dev_path_);
    if (!setup_and_detect())
      throw std::runtime_error("CH347 LSM6DS3 not detected on 0x6A/0x6B");
  }

private:
  // 打开后的统一配置 + IMU 探测；成功返回 true（addr_ 已就绪）
  bool setup_and_detect() {
    if (fd_ < 0) return false;
    // 超时设置在新库上可能签名不同或缺符号：失败仅告警不致命
    if (!set_timeout_(fd_, 2000, 2000))
      LOGW("CH34xSetTimeout failed (tolerated; some lib builds don't need it)");
    if (!i2c_set_(fd_, 0x01)) return false;  // 0x01 = 400kHz I2C
    if (i2c_set_ignore_nack_) i2c_set_ignore_nack_(fd_, 1);
    if (i2c_set_stretch_) i2c_set_stretch_(fd_, true);
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
              return true;
            }
          }
        } catch (...) { continue; }
      }
      util::sleep_for(20);
    }
    return false;
  }

public:

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
    std::vector<uint8_t> rbuf(length);
    if (stream_i2c_ret_ack_) {
      uint8_t addr_write[] = {static_cast<uint8_t>(addr_ << 1 | 0x00), start_reg};
      int ack = 0;
      if (!stream_i2c_ret_ack_(fd_, 2, addr_write, 0, nullptr, &ack))
        throw std::runtime_error("CH347 I2C block write address failed");
      uint8_t addr_read[] = {static_cast<uint8_t>(addr_ << 1 | 0x01)};
      if (!stream_i2c_ret_ack_(fd_, 1, addr_read, length, rbuf.data(), &ack))
        throw std::runtime_error("CH347 I2C block read data failed");
    } else {
      // Older lib without RetAck: combined write+read in one stream call.
      uint8_t buf[2] = {static_cast<uint8_t>(addr_ << 1), start_reg};
      if (!stream_i2c_(fd_, 2, buf, length, rbuf.data()))
        throw std::runtime_error("CH347 I2C block read failed");
    }
    return rbuf;
  }

  cereal::SensorEventData::SensorSource source() const override { return source_; }

private:
  std::vector<uint8_t> stream_read(const uint8_t *write_bytes, int write_len, int read_len) {
    std::vector<uint8_t> rbuf(read_len);
    if (stream_i2c_ret_ack_) {
      int ack = 0;
      if (!stream_i2c_ret_ack_(fd_, write_len, const_cast<uint8_t *>(write_bytes),
                               read_len, rbuf.data(), &ack))
        throw std::runtime_error("CH347 I2C read failed");
    } else {
      if (!stream_i2c_(fd_, write_len, const_cast<uint8_t *>(write_bytes),
                       read_len, rbuf.data()))
        throw std::runtime_error("CH347 I2C read failed");
    }
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
  std::string project_root = get_project_root();

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

// Check whether a /dev/hidrawN node belongs to a WCH chip in multi-function
// (SPI/I2C) mode: HID ID must be 1a86:55db or 1a86:55dc. Optionally skips the
// HID-UART interface so we land on the SPI+I2C+GPIO one.
static bool hidraw_is_ch347_i2c(const char *dev_path, bool allow_uart_iface) {
  const char *slash = std::strrchr(dev_path, '/');
  if (!slash) return false;
  std::string base(slash + 1);

  // Resolve HID ID via sysfs: /sys/class/hidraw/<base>/device/uevent
  std::ifstream f("/sys/class/hidraw/" + base + "/device/uevent");
  if (!f.is_open()) return false;
  std::string line;
  unsigned int vid = 0, pid = 0;
  bool have_id = false;
  while (std::getline(f, line)) {
    if (line.rfind("HID_ID=", 0) == 0) {
      unsigned int bus = 0;
      if (std::sscanf(line.c_str(), "HID_ID=%x:%x:%x", &bus, &vid, &pid) == 3)
        have_id = true;
      break;
    }
  }
  if (!have_id) return false;
  if (vid != 0x1A86 || (pid != 0x55DB && pid != 0x55DC)) return false;

  // Distinguish the two HID interfaces when possible: the USB interface
  // name of the UART one usually mentions UART/Serial/COM.
  std::ifstream iface_file("/sys/class/hidraw/" + base + "/device/../interface");
  if (iface_file.is_open()) {
    std::string iface((std::istreambuf_iterator<char>(iface_file)),
                       std::istreambuf_iterator<char>());
    for (char &c : iface) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    bool looks_uart = iface.find("UART") != std::string::npos ||
                      iface.find("SERIAL") != std::string::npos ||
                      iface.find("COM") != std::string::npos;
    if (looks_uart && !allow_uart_iface) return false;
  }
  return true;
}

static bool detect_ch347_backend(std::string &dev_path, std::string &lib_path) {
  const char *env_dev = std::getenv("SENSORD_CH347_DEV");
  if (env_dev && env_dev[0] != '\0') {
    dev_path = env_dev;
  } else {
    glob_t globbuf;
    bool found = false;
    // 1) Vendor kernel driver node (ch34x_pis.ko loaded)
    if (glob("/dev/ch34x_pis*", 0, nullptr, &globbuf) == 0 && globbuf.gl_pathc > 0) {
      dev_path = globbuf.gl_pathv[0];
      found = true;
    }
    globfree(&globbuf);
    // 2) Stock HID driver: CH347T in mode 2 exposes SPI+I2C as /dev/hidrawN
    if (!found && glob("/dev/hidraw*", GLOB_NOSORT, nullptr, &globbuf) == 0) {
      for (size_t i = 0; i < globbuf.gl_pathc && !found; i++) {
        if (hidraw_is_ch347_i2c(globbuf.gl_pathv[i], /*allow_uart_iface=*/false)) {
          dev_path = globbuf.gl_pathv[i];
          found = true;
        }
      }
      // Fall back to any 1a86:55db hidraw if none matched the name filter
      for (size_t i = 0; i < globbuf.gl_pathc && !found; i++) {
        if (hidraw_is_ch347_i2c(globbuf.gl_pathv[i], /*allow_uart_iface=*/true)) {
          dev_path = globbuf.gl_pathv[i];
          found = true;
        }
      }
      globfree(&globbuf);
      if (found) LOG("Using CH347 in HID multi-function mode (%s), no kernel module needed", dev_path.c_str());
    }
    // 3) Last resort: CDC ACM (dual-serial mode 0 - cannot reach the IMU,
    //    kept only so existing setups keep working)
    if (!found) {
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
static bool bus_forbidden(int bus) {
  for (int b : FORBIDDEN_I2C_BUSES) {
    if (b == bus) return true;
  }
  return false;
}

static int detect_imu_i2c_bus(uint8_t &addr) {
  // 仅当显式设置 SENSORD_I2C_BUS 时才探测该总线（默认完全不探测本地 I2C，
  // 本机实测：任何未接设备的空总线裸写都会触发超时风暴，见文件头注释）
  const char *env_bus = std::getenv("SENSORD_I2C_BUS");
  if (!env_bus || env_bus[0] == '\0') {
    LOG("SENSORD_I2C_BUS not set: direct-I2C probing disabled by safety policy (CH347 USB only)");
    return -1;
  }
  {
    int bus = atoi(env_bus);
    if (bus < 0 || bus_forbidden(bus)) {
      LOGW("SENSORD_I2C_BUS=%d rejected: system-critical bus (safety, see IMU/ch347t_安全修复说明.md)", bus);
      return -1;
    }
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
    return -1;
  }
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
  bool gyro_bias_done = false;
  uint64_t auto_bias_t0 = 0;
  double bias_sum[3] = {0.0, 0.0, 0.0};
  double bias_sumsq[3] = {0.0, 0.0, 0.0};
  int bias_n = 0;

  try {
    sensor.open();
    sensor.init_sensor();
    LOG("Using %s backend", backend_name.c_str());
  } catch (const std::exception &e) {
    LOGE("%s: open/init failed: %s", backend_name.c_str(), e.what());
    return false;
  }

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
        cache.gyro_v[0] = y * GYRO_SCALE;
        cache.gyro_v[1] = -x * GYRO_SCALE;
        cache.gyro_v[2] = z * GYRO_SCALE;
        fresh_gyro = true;
      } catch (...) { i2c_error = true; }
    }

    // 校准应用：gyro 减零偏(自动零偏优先，json 作初值)；accel 减偏置乘矩阵(json)
    if (!i2c_error && (fresh_acc || fresh_gyro)) {
      // --- 启动自动零偏校准：累积 ~5s 求均值/标准差，1σ 小 = 静止（零偏本身再大也不影响） ---
      // 前 1s 跳过(传感器配置稳定期)；std>0.015 rad/s 视为运动/振动，拒绝并沿用 json/零偏。
      // 零偏随温度漂移，每次开机重测。注意不可用"瞬时模长<阈值"判静止——零偏会撑爆阈值。
      if (!gyro_bias_done && fresh_gyro) {
        if (auto_bias_t0 == 0) auto_bias_t0 = nanos_since_boot();
        if (nanos_since_boot() - auto_bias_t0 >= 1e9) {  // 跳过开头 1s
          for (int i = 0; i < 3; i++) {
            bias_sum[i] += cache.gyro_v[i];
            bias_sumsq[i] += cache.gyro_v[i] * cache.gyro_v[i];
          }
          bias_n++;
        }
        if (bias_n >= 500) {  // ~5s @104Hz
          gyro_bias_done = true;
          float max_std = 0.f;
          for (int i = 0; i < 3; i++) {
            float mean = (float)(bias_sum[i] / bias_n);
            float var = (float)(bias_sumsq[i] / bias_n) - mean * mean;
            if (var < 0.f) var = 0.f;
            max_std = std::max(max_std, std::sqrt(var));
          }
          if (max_std < 0.015f) {  // 1σ < ~0.86°/s = 静止
            for (int i = 0; i < 3; i++)
              g_runtime_gyro_bias_rad[i] = (float)(bias_sum[i] / bias_n);
            LOGW("gyro auto zero-bias calibrated: [%.5f, %.5f, %.5f] rad/s (1sigma=%.5f, %d samples)",
                 g_runtime_gyro_bias_rad[0], g_runtime_gyro_bias_rad[1], g_runtime_gyro_bias_rad[2], max_std, bias_n);
          } else {
            LOGW("gyro auto zero-bias rejected: not stationary (1sigma=%.5f rad/s), using json/zero bias", max_std);
          }
        } else if (nanos_since_boot() - auto_bias_t0 > 12e9) {
          gyro_bias_done = true;
          LOGW("gyro auto zero-bias skipped: insufficient samples within 12s (using json/zero bias)");
        }
      }
      for (int i = 0; i < 3; i++)
        cache.gyro_v[i] -= g_runtime_gyro_bias_rad[i];
      if (g_imu_calib.valid) {
        float acc_raw[3] = {
          cache.acc_v[0] - 9.81f * g_imu_calib.accel_bias_g[0],
          cache.acc_v[1] - 9.81f * g_imu_calib.accel_bias_g[1],
          cache.acc_v[2] - 9.81f * g_imu_calib.accel_bias_g[2]};
        for (int i = 0; i < 3; i++)
          cache.acc_v[i] = g_imu_calib.matrix[i * 3] * acc_raw[0]
                         + g_imu_calib.matrix[i * 3 + 1] * acc_raw[1]
                         + g_imu_calib.matrix[i * 3 + 2] * acc_raw[2];
      }
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
  // libch347.so 每个 I2C 事务都往 stdout printf 调试行（mLength: .. iReadLength: .. AckBitCnt: ..），
  // 日志量巨大且无信息量 → 重定向到 /dev/null（2026-08-30；auto_calibrate.py 早已用同款手法，见交接文档 §七）
  if (!std::getenv("CH347_LIB_DEBUG")) {
    FILE *devnull = std::fopen("/dev/null", "w");
    if (devnull) { std::freopen("/dev/null", "w", stdout); std::fclose(devnull); }
  }
  PubMaster publisher({"accelerometer", "gyroscope", "temperatureSensor"});
  std::atomic<bool> switch_backend{false};

  // 可选 IMU 校准参数：IMU_CALIB_JSON 环境变量优先，默认仓库根 imu_calibration.json
  const char *env_calib = std::getenv("IMU_CALIB_JSON");
  std::string calib_path = (env_calib && env_calib[0] != '\0')
      ? std::string(env_calib) : get_project_root() + "/imu_calibration.json";
  load_imu_calibration(calib_path);

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
    if (i2c_bus < 0) {
      // 没有任何安全总线可探测：不碰系统总线，等 CH347 插入（manager watchdog 会拉起）
      LOGW("No safe direct-I2C bus available; waiting for CH347 USB device...");
      util::sleep_for(CH347_RECHECK_INTERVAL_MS);
      continue;
    }
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
