"""Validate button semantics and the complete Cartesian motion before physics."""

import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pressb.kinematics import PiperKinematics
from pressb.planning import base_position, button_centers, make_plan


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/scene.json").read_text())
URDF = ROOT / CFG["robot_urdf"]


class ButtonNumberingTests(unittest.TestCase):
    def test_two_columns_increase_from_bottom_to_top(self):
        centers = button_centers(CFG)
        self.assertEqual(set(centers), set(range(24, 36)))
        for row in range(6):
            left, right = centers[24 + row], centers[30 + row]
            self.assertGreater(left[1], right[1])  # Looking +X, +Y is left.
            self.assertEqual(left[2], right[2])
            self.assertAlmostEqual(left[2], CFG["button_bottom_z"] + row * CFG["button_pitch_z"])
            self.assertAlmostEqual(left[0], CFG["button_face_x"])


@unittest.skipUnless(URDF.is_file(), "Download the official Piper asset first")
class CompleteTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = PiperKinematics(URDF, tip_offset=CFG["tip_offset"])
        cls.plan = make_plan(cls.kin, CFG)
        cls.poses = np.array([cls.kin.fk(q) for q in cls.plan.q])
        cls.positions = cls.poses[:, :3, 3].copy()
        cls.positions += base_position(CFG)

    def test_all_twelve_buttons_have_press_hold_and_retract(self):
        centers = button_centers(CFG)
        for floor in range(24, 36):
            for phase in ("press", "hold", "retract"):
                self.assertTrue(np.any((self.plan.floor == floor) & (self.plan.phase == phase)))
            mask = (self.plan.floor == floor) & (self.plan.phase == "hold")
            expected = centers[floor] + [CFG["press_depth"], 0, 0]
            np.testing.assert_allclose(self.positions[mask], np.tile(expected, (mask.sum(), 1)), atol=5e-4)

    def test_motion_stays_within_joint_range_and_speed(self):
        self.assertTrue(np.all(self.plan.q >= self.kin.lower))
        self.assertTrue(np.all(self.plan.q <= self.kin.upper))
        speed = np.abs(np.diff(self.plan.q, axis=0)) / CFG["physics_dt"]
        self.assertLessEqual(np.max(speed), CFG["joint_speed"])
        self.assertLess(np.max(np.abs(np.diff(self.plan.q, axis=0))), 0.01)
        np.testing.assert_allclose(np.diff(self.plan.time), CFG["physics_dt"], atol=1e-12)

    def test_cartesian_tracking_and_tool_orientation(self):
        self.assertLess(np.max(np.linalg.norm(self.positions - self.plan.target_tip, axis=1)), 5e-4)
        desired = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        pressing = np.isin(self.plan.phase, ["press", "hold", "retract"])
        errors = Rotation.from_matrix(desired @ np.transpose(self.poses[pressing, :3, :3], (0, 2, 1))).magnitude()
        self.assertLess(np.max(errors), 1e-2)

    def test_transfers_remain_behind_the_panel(self):
        mask = np.isin(self.plan.phase, ["depart_home", "approach", "return_home"])
        sphere_front_x = self.positions[mask, 0] + .005 * (1. - self.poses[mask, 0, 2])
        self.assertLess(np.max(sphere_front_x), CFG["button_face_x"] - .005)

    def test_home_is_the_official_stacked_fold_above_the_table(self):
        frames = self.kin.link_transforms(CFG["home_q"])
        shoulder, elbow, wrist = (frames[name][:3, 3] for name in ("link2", "link3", "link4"))
        upper, forearm = elbow - shoulder, wrist - elbow
        # Official piper_urdf_zero.png shows the long dogleg housings stacked.
        # Their offset joint centers slope ~8/10 degrees; requiring those center
        # lines to be exactly level would incorrectly select an extended pose.
        self.assertLess(upper[0], -.20)
        self.assertGreater(forearm[0], .20)
        for vector in (upper, forearm):
            self.assertLess(abs(vector[1]), 1e-5)
            self.assertLess(abs(np.arctan2(vector[2], abs(vector[0]))), np.deg2rad(12))
        overlap = min(shoulder[0], wrist[0]) - elbow[0]
        self.assertGreater(overlap, .20)
        self.assertGreater(elbow[2], shoulder[2])
        self.assertGreater(wrist[2], elbow[2])
        self.assertGreater(min(shoulder[2], elbow[2], wrist[2]), .10)
        np.testing.assert_allclose(CFG["home_q"], [0., .069762054, 0., 0., 0., 0.], atol=1e-9)
        self.assertAlmostEqual(CFG["table_edge_x"] - CFG["robot_base_x"], .10)

    def test_home_levels_the_actual_casings_within_the_official_elbow_limit(self):
        # This uses measured STL casing axes, independently of the dogleg joint
        # centers. Splitting their residual mismatch gives ~1 degree per casing.
        tilts = self.kin.housing_tilts_deg(CFG["home_q"])
        self.assertLess(np.max(np.abs(tilts)), 1.01)
        self.assertAlmostEqual(tilts[0], tilts[1], places=5)
        self.assertAlmostEqual(CFG["home_q"][2], self.kin.upper[2])
        nominal = self.kin.housing_tilts_deg(np.zeros(6))
        self.assertLess(np.max(np.abs(tilts)), np.max(np.abs(nominal)))

    def test_every_press_returns_to_identical_home_before_the_next_cycle(self):
        home = np.asarray(CFG["home_q"])
        np.testing.assert_allclose(self.plan.q[0], home, atol=1e-12)
        np.testing.assert_allclose(self.plan.q[-1], home, atol=1e-12)
        for index, floor in enumerate(CFG["sequence"]):
            hold_indices = np.flatnonzero((self.plan.floor == floor) & (self.plan.phase == "home_hold"))
            self.assertGreaterEqual(len(hold_indices), int(CFG["home_hold_duration"] / CFG["physics_dt"]))
            np.testing.assert_allclose(self.plan.q[hold_indices], np.tile(home, (len(hold_indices), 1)), atol=1e-12)
            retract_indices = np.flatnonzero((self.plan.floor == floor) & (self.plan.phase == "retract"))
            self.assertGreater(hold_indices[0], retract_indices[-1])
            if index + 1 < len(CFG["sequence"]):
                next_cycle = np.flatnonzero(self.plan.floor == CFG["sequence"][index + 1])
                self.assertLess(hold_indices[-1], next_cycle[0])


if __name__ == "__main__":
    unittest.main()
