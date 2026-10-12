"""Chinese-friendly labels for the eGPU settings panel.

`usbgpu_status()` and `check_usbgpu()` return stable English tokens (they are
also compared against in tests), so translate at the point of display instead of
changing what they return.
"""
from openpilot.system.ui.lib.multilang import tr, tr_noop

# Tokens returned by usbgpu_status()/check_usbgpu(). They are translated through
# tr() at display time, so declare them here for the .pot extractor.
TRANSLATED_TOKENS = [
  tr_noop("not detected"),
  tr_noop("multiple devices"),
  tr_noop("firmware mismatch"),
  tr_noop("startup failed"),
  tr_noop("loading"),
  tr_noop("reboot to compile"),
  tr_noop("active"),
  tr_noop("model not compiled"),
  tr_noop("ready"),
  tr_noop("USB not connected"),
  tr_noop("not connected"),
  tr_noop("not checked"),
  tr_noop("checking..."),
  tr_noop("GPU check timed out"),
  tr_noop("12V / PCIe not ready"),
  tr_noop("GPU incompatible"),
  tr_noop("USB disconnected during GPU check"),
  tr_noop("USB link errors"),
]


def tr_usbgpu_status(status: str) -> str:
  if status.startswith("slow USB ("):
    return tr("slow USB ({} Mbps)").format(status.removeprefix("slow USB (").removesuffix(" Mbps)"))
  return tr(status)


def tr_usbgpu_check(result: str | None) -> str:
  if result is None:
    return tr("no errors")
  if result.startswith("USB link ") and result.endswith(" Mbps"):
    return tr("USB link {} Mbps").format(result.removeprefix("USB link ").removesuffix(" Mbps"))
  if result.startswith("power status unavailable: "):
    return tr("power status unavailable: {}").format(result.removeprefix("power status unavailable: "))
  if result.startswith("eGPU power fault (") and result.endswith(" mA)"):
    detail = result.removeprefix("eGPU power fault (").removesuffix(" mA)")
    voltage, _, current = detail.partition(", ")
    return tr("eGPU power fault ({} mV, {} mA)").format(voltage, current)
  if result.startswith("12V off ("):
    return tr("12V off ({} mV)").format(result.removeprefix("12V off (").removesuffix(" mV)"))
  return tr(result)