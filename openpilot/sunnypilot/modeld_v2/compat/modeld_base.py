# Compat base class for ModelState
# Minimal implementation that old-layout forks can use

import numpy as np


class ModelStateBase:
  def __init__(self):
    pass

  def run(self, bufs, transforms, inputs, prepare_only):
    raise NotImplementedError

  def get_action_from_model(self, model_output, prev_action, lat_action_t, long_action_t, v_ego):
    raise NotImplementedError
