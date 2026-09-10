# Compat parser for split model outputs (vision + policy separate)

import numpy as np


class Parser:
  def parse_vision_outputs(self, sliced: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outputs = {}
    for key, val in sliced.items():
      outputs[key] = val
    return outputs

  def parse_policy_outputs(self, sliced: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outputs = {}
    for key, val in sliced.items():
      outputs[key] = val
    return outputs
