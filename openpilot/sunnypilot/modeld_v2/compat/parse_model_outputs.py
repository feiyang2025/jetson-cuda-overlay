# Compat parser for merged model outputs
# Parses combined vision+policy output

import numpy as np


class Parser:
  def parse_outputs(self, sliced: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    outputs = {}
    for key, val in sliced.items():
      outputs[key] = val
    return outputs
