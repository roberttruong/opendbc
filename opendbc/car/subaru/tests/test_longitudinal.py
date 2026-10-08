import itertools
import unittest

import numpy as np

from opendbc.car.subaru.longitudinal import ACCEL_BP, RPM, RPM_STANDSTILL, SPEED_BP, THROTTLE, get_long_command
from opendbc.car.subaru.values import CarControllerParams as P

SPEEDS = np.arange(0., 40.1, 1.)
ACCELS = np.arange(-4., 3.01, 0.05)
HILLS = [-1., -0.3, 0., 0.3, 1.]


class TestSubaruLongitudinal(unittest.TestCase):
  def test_table_shape(self):
    for table in (THROTTLE, RPM):
      assert len(table) == len(SPEED_BP)
      assert all(len(row) == len(ACCEL_BP) for row in table)

  def test_within_safety_limits(self):
    for v, a, hill, braking_prev in itertools.product(SPEEDS, ACCELS, HILLS, (False, True)):
      throttle, rpm, brake, _ = get_long_command(a, hill, v, braking_prev)
      assert P.THROTTLE_MIN <= throttle <= P.THROTTLE_MAX
      assert P.RPM_MIN <= rpm <= P.RPM_MAX
      assert P.BRAKE_MIN <= brake <= P.BRAKE_MAX

  def test_throttle_and_brake_exclusive(self):
    for v, a in itertools.product(SPEEDS, ACCELS):
      throttle, _, brake, braking = get_long_command(a, 0., v, False)
      if braking:
        assert throttle == P.THROTTLE_ENGINE_BRAKE
      else:
        assert brake == 0

  def test_monotonic_in_accel(self):
    # asking for more accel never gives less throttle, RPM or more brake
    for v in SPEEDS:
      prev = None
      for a in ACCELS:
        throttle, rpm, brake, braking = get_long_command(a, 0., v, False)
        if prev is not None and not braking and not prev[3]:
          assert throttle >= prev[0] and rpm >= prev[1]
        if prev is not None and braking and prev[3]:
          assert brake <= prev[2]
        prev = (throttle, rpm, brake, braking)

  def test_hill_adds_throttle(self):
    flat, *_ = get_long_command(0., 0., 30., False)
    uphill, *_ = get_long_command(0., 0.3, 30., False)  # ~3% grade
    assert uphill > flat + 300

  def test_highway_cruise(self):
    # EyeSight holds ~67 mph on flat road with ~2680 throttle and ~1870 rpm
    throttle, rpm, brake, braking = get_long_command(0., 0., 30., False)
    assert not braking and brake == 0
    assert abs(throttle - 2680) < 20 and abs(rpm - 1870) < 20

  def test_hard_braking_saturates(self):
    _, _, brake, braking = get_long_command(-3.0, 0., 20., False)
    assert braking and brake == P.BRAKE_MAX
    _, _, brake, _ = get_long_command(-2.0, 0., 20., False)
    assert 0 < brake < P.BRAKE_MAX

  def test_brake_hysteresis(self):
    # just above the 30 m/s brake threshold (-0.53): stays braking only if already braking
    assert not get_long_command(-0.51, 0., 30., False)[3]
    assert get_long_command(-0.51, 0., 30., True)[3]

  def test_standstill_hold(self):
    throttle, rpm, brake, braking = get_long_command(-0.5, 0., 0., True)
    assert braking and throttle == P.THROTTLE_ENGINE_BRAKE and rpm == RPM_STANDSTILL
    assert 100 < brake < 150  # EyeSight holds the car with ~117


if __name__ == "__main__":
  unittest.main()
