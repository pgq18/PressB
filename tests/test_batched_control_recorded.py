"""Small static regression from a successful near-home press demonstration.

Targets are recorded base-frame TCP poses; this fixture needs no dataset or
outputs directory. It stresses joint2/joint3 boundaries and the near-singular
wrist during departure from home, not only nonsingular random configurations.
"""
import json
from pathlib import Path

import numpy as np

from pressb.online_rl.batched_control import BatchedPoseController

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / json.loads((ROOT / "configs/scene.json").read_text())["robot_urdf"]
# 5090_smoke_30hz_v2 episode 0, frames 28 through 37; immutable numeric fixture.
INITIAL_Q = np.array([0.0002730209962464869, 0.07370296120643616, -0.001260218909010291, 0.001119325403124094,
 -0.001810774439945817, -0.000256091560004279])
TARGETS = np.array([[0.19854469522086504, 8.663833210620386e-05, 0.2124811416357926, 0.7132319930695246,
  -8.601839819614022e-05, 0.700928004499186, 0.0002217469225820167, 0.008],
 [0.19864776130651324, 0.00010557813317705369, 0.21270750538591893, 0.7132288886580569,
  -0.00010489494086912548, 0.7009311437305218, 0.0002704242350961155, 0.008],
 [0.19876447230090627, 0.00012690255711768334, 0.2129624905162243, 0.7132253854679099,
  -0.00012609729954177868, 0.7009346815916435, 0.0003252163029691502, 0.008],
 [0.19889563426312276, 0.00015071422110225918, 0.21324736467853084, 0.7132214637652298,
  -0.00014970868139338887, 0.7009386363453188, 0.0003863809761121683, 0.008],
 [0.19904205985145038, 0.00017710896690460078, 0.2135633197776324, 0.7132171044259016,
  -0.00017580206765113295, 0.7009430253516632, 0.000454157267158423, 0.008],
 [0.1992045712324351, 0.00020617587295739364, 0.21391147167401448, 0.7132122889049733,
  -0.00020443994892328234, 0.7009478650846059, 0.0005287651363539085, 0.008],
 [0.1993840028567562, 0.00023799728334862006, 0.21429285985345906, 0.7132069992069943,
  -0.0002356740822159412, 0.7009531711500188, 0.0006104052866231945, 0.008],
 [0.1995812040812487, 0.0002726488555888278, 0.21470844706325662, 0.7132012178574084,
  -0.0002695452727224862, 0.7009589583056537, 0.0006992589708893092, 0.008],
 [0.19979704161798384, 0.0003101996288284966, 0.21515911891471337, 0.7131949278751347,
  -0.0003060831832185064, 0.7009652404830106, 0.000795487813540298, 0.008],
 [0.20003240179287235, 0.0003507121140334733, 0.21564568345173568, 0.7131881127464529,
  -0.00034530617363849516, 0.7009720308112486, 0.0008992336477662864, 0.008]])


def test_recorded_near_home_departure_tracks_full_pose_on_same_bounded_branch():
    controller = BatchedPoseController(URDF, INITIAL_Q[None])
    previous = controller.q.copy()
    for target in TARGETS:
        q, diagnostics = controller.solve([0], target[None])
        assert np.all(np.abs(q - previous) <= controller.maximum_joint_step + 1e-10)
        assert np.all(q >= controller.kinematics.lower)
        assert np.all(q <= controller.kinematics.upper)
        # Sub-micrometre fits on an actual near-singular departure trajectory.
        assert diagnostics[0]["command_position_residual_m"] < 1e-6
        assert diagnostics[0]["command_rotation_residual_rad"] < 1e-5
        previous = q
    # The target moves out of home; a solver that stalls or swaps wrist branches
    # can fit easy static poses but must not pass this sequence.
    assert q[0, 1] > INITIAL_Q[1] + .015
    assert np.max(np.abs(q[0, [0, 3, 5]])) < .02
