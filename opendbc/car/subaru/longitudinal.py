import numpy as np

from opendbc.car.subaru.values import CAR, CarControllerParams

# same FB20 engine and TR580 CVT; the data the tables were fitted on
LONG_TABLE_CARS = {CAR.SUBARU_IMPREZA, CAR.SUBARU_IMPREZA_2020}

# Feedforward fitted to stock EyeSight commands from 22 Impreza 2017-22 cars (2.5 h of ACC driving).
# Both tables are indexed by speed and (requested accel + road grade), and are bilinearly interpolated.
SPEED_BP = [0., 5., 10., 15., 20., 25., 30., 35.]   # m/s
ACCEL_BP = [-1.0, -0.5, 0., 0.25, 0.5, 1.0, 1.5, 2.0]  # m/s^2, requested accel + grade

THROTTLE = [
  [1520, 1660, 1810, 1960, 2130, 2310, 2500, 2700],
  [1460, 1700, 1940, 2180, 2410, 2640, 2870, 3090],
  [1400, 1740, 2070, 2400, 2690, 2960, 3220, 3400],
  [1320, 1770, 2210, 2620, 2970, 3270, 3400, 3400],
  [1230, 1810, 2380, 2870, 3230, 3400, 3400, 3400],
  [1140, 1830, 2530, 3100, 3400, 3400, 3400, 3400],
  [1060, 1860, 2680, 3300, 3400, 3400, 3400, 3400],
  [1010, 1950, 2860, 3400, 3400, 3400, 3400, 3400],
]

# Cruise_RPM is an engine speed target that the CVT follows
RPM = [
  [750, 800, 840, 850, 850, 850, 850, 850],
  [710, 840, 960, 1060, 1150, 1260, 1380, 1510],
  [690, 880, 1070, 1260, 1450, 1680, 1920, 2160],
  [690, 930, 1190, 1460, 1760, 2100, 2460, 2800],
  [750, 1020, 1310, 1690, 2110, 2550, 2990, 3430],
  [880, 1190, 1530, 1990, 2510, 3020, 3500, 3500],
  [1060, 1470, 1870, 2390, 2950, 3500, 3500, 3500],
  [1270, 1840, 2380, 2900, 3440, 3500, 3500, 3500],
]

# EyeSight brakes (with throttle at THROTTLE_ENGINE_BRAKE) once accel + grade drops below this, which gets
# lower with speed as drag does more of the slowing. Fitted as the point it brakes in half the samples.
BRAKE_START_BP = [0., 7.5, 12.5, 17.5, 22.5, 27.5, 33.]  # m/s
BRAKE_START_V = [-0.1, -0.1, -0.25, -0.3, -0.4, -0.5, -0.55]  # m/s^2
BRAKE_HYSTERESIS = 0.05  # m/s^2

# decel + grade ~= BRAKE_GAIN * Brake_Pressure + BRAKE_DRAG * vEgo + BRAKE_OFFSET
BRAKE_GAIN = 0.0047     # m/s^2 per count
BRAKE_DRAG = 0.0111     # m/s^2 per m/s
BRAKE_OFFSET = -0.088   # m/s^2

RPM_STANDSTILL = 100  # what EyeSight sends while holding the car stopped


def interp2d(x: float, y: float, x_bp: list[float], y_bp: list[float], table: list[list[int]]) -> float:
  """Bilinear interpolation of table[x_index][y_index], clamped to the breakpoints."""
  column = [np.interp(y, y_bp, row) for row in table]
  return float(np.interp(x, x_bp, column))


def get_long_command(accel: float, hill: float, v_ego: float, braking_prev: bool) -> tuple[int, int, int, bool]:
  """Returns (Cruise_Throttle, Cruise_RPM, Brake_Pressure, braking) for a requested accel."""
  v_ego = max(v_ego, 0.)
  target = accel + hill

  brake_start = float(np.interp(v_ego, BRAKE_START_BP, BRAKE_START_V))
  braking = target < (brake_start + BRAKE_HYSTERESIS if braking_prev else brake_start)

  if braking:
    throttle = CarControllerParams.THROTTLE_ENGINE_BRAKE
    brake = (-target - BRAKE_DRAG * v_ego - BRAKE_OFFSET) / BRAKE_GAIN
    # hold the CVT ratio needed to cruise at this speed, like EyeSight
    rpm = interp2d(v_ego, 0., SPEED_BP, ACCEL_BP, RPM) if v_ego > 0.1 else RPM_STANDSTILL
  else:
    throttle = interp2d(v_ego, target, SPEED_BP, ACCEL_BP, THROTTLE)
    rpm = interp2d(v_ego, target, SPEED_BP, ACCEL_BP, RPM)
    brake = CarControllerParams.BRAKE_MIN

  throttle = int(np.clip(round(throttle), CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX))
  rpm = int(np.clip(round(rpm), CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX))
  brake = int(np.clip(round(brake), CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX))
  return throttle, rpm, brake, braking
