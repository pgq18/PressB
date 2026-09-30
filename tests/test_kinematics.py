"""Check kinematics against the official URDF, including impossible targets."""

from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf"


@unittest.skipUnless(URDF.is_file(), "Download the official Piper asset first")
class PiperKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.kin = PiperKinematics(URDF, tip_offset=0.12)

    def test_joint_limits_and_tool_axis(self):
        self.assertEqual(self.kin.joint_names, [f"joint{i}" for i in range(1, 7)])
        q = np.array([0.1, 1.0, -1.0, 0.2, 0.3, -0.2])
        flange = PiperKinematics(URDF, tip_offset=0.0).fk(q)
        tool = self.kin.fk(q)
        np.testing.assert_allclose(tool[:3, 3] - flange[:3, 3], flange[:3, 2] * 0.12, atol=1e-12)
        np.testing.assert_allclose(tool[:3, :3].T @ tool[:3, :3], np.eye(3), atol=1e-12)

    def test_full_pose_inverse_kinematics(self):
        desired = self.kin.fk(np.array([0.2, 1.1, -1.4, 0.2, 0.4, -0.3]))
        solved = self.kin.solve(desired[:3, 3], np.array([0., 1., -1., 0., 0.5, 0.]), desired[:3, :3])
        achieved = self.kin.fk(solved)
        self.assertLess(np.linalg.norm(achieved[:3, 3] - desired[:3, 3]), 5e-4)
        self.assertLess(Rotation.from_matrix(desired[:3, :3] @ achieved[:3, :3].T).magnitude(), 1e-2)
        self.assertTrue(np.all(solved >= self.kin.lower))
        self.assertTrue(np.all(solved <= self.kin.upper))

    def test_position_only_inverse_kinematics(self):
        desired = self.kin.fk(np.array([-0.15, 1.3, -1.5, 0.1, 0.6, 0.]))[:3, 3]
        solved = self.kin.solve(desired, np.array([0., 1., -1., 0., 0.5, 0.]))
        self.assertLess(np.linalg.norm(self.kin.fk(solved)[:3, 3] - desired), 5e-4)

    def test_unreachable_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unreachable"):
            self.kin.solve([10., 0., 0.], [0., 1., -1., 0., 0., 0.])

    def test_invalid_orientation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rotation"):
            self.kin.solve([0.3, 0., 0.3], [0., 1., -1., 0., 0., 0.], np.zeros((3, 3)))

if __name__ == "__main__":
    unittest.main()
