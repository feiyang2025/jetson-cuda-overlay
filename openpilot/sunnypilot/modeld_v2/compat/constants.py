# Compat constants for old-layout openpilot forks
# Provides ModelConstants and Plan that gpu_model_state.py needs


class ModelConstants:
  MODEL_RUN_FREQ = 20
  MODEL_CONTEXT_FREQ = 5
  N_FRAMES = 5
  DESIRE_LEN = 32
  T_IDXS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  PLAN_WIDTH = 4


class Plan:
  VELOCITY = 0
  ACCELERATION = 1
  T_FROM_CURRENT_EULER = 2
  ORIENTATION_RATE = 3
