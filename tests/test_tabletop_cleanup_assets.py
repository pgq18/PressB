"""The authored shapes, physical holes and containment bounds must agree."""
from collections import Counter
import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from pressb.tasks.tabletop_cleanup.geometry import (
    KINDS, SPEAKER_EAR_CENTER_XY, asset_geometry, asset_spec,
)


def _volume(part):
    v=np.asarray(part.vertices)
    return sum(np.dot(v[f[0]],np.cross(v[f[i]],v[f[i+1]]))
               for f in part.faces for i in range(1,len(f)-1))/6


@pytest.mark.parametrize("kind",KINDS)
def test_meshes_closed_and_compound_collisions_accurate(kind):
    parts,spec=asset_geometry(kind)
    assert len(parts)>1 and spec["num_collision_parts"]>1
    for part in parts:
        edges=Counter(tuple(sorted((a,b))) for f in part.faces for a,b in zip(f,f[1:]+f[:1]))
        assert set(edges.values())=={2},part.name
        assert _volume(part)>0,part.name
        if part.collision:
            # Curved gamepad grips differ from their convex cooks by 0.11%;
            # all other compound pieces are already convex by construction.
            assert ConvexHull(part.vertices).volume/_volume(part)<1.002,part.name
    visual=np.concatenate([np.asarray(p.vertices) for p in parts if p.material!="collision"])
    support=np.asarray(spec["support_vertices"])
    np.testing.assert_allclose(visual.max(0)-visual.min(0),spec["size"],atol=1e-12)
    np.testing.assert_allclose(visual.min(0),-np.asarray(spec["size"])/2,atol=1e-12)
    # The finite support set must preserve extrema after arbitrary rotation.
    from scipy.spatial.transform import Rotation
    for rotation in Rotation.random(5,random_state=7).as_matrix():
        np.testing.assert_allclose((visual@rotation.T).max(0),(support@rotation.T).max(0),atol=1e-10)
        np.testing.assert_allclose((visual@rotation.T).min(0),(support@rotation.T).min(0),atol=1e-10)


def _inside_convex(vertices,point,tolerance=1e-10):
    equations=ConvexHull(vertices).equations
    return np.all(equations[:,:3]@point+equations[:,3]<=tolerance)


def test_speaker_measured_enclosure_is_not_confused_with_ears_and_holes_are_empty():
    parts,spec=asset_geometry("speaker")
    np.testing.assert_allclose(spec["measured_body_size"],[.054,.039,.023])
    np.testing.assert_allclose(spec["size"],[.067,.049,.02332])
    assert spec["grasp_candidates"][0]["surface_width_m"]==.039
    # A full ray through each ear centre must remain outside every cooked
    # collision hull, including the main enclosure and the top cover.
    for sx in (-1,1):
        for sy in (-1,1):
            for z in np.linspace(-spec["size"][2]/2,spec["size"][2]/2,9):
                point=np.array([sx*SPEAKER_EAR_CENTER_XY[0],sy*SPEAKER_EAR_CENTER_XY[1],z])
                assert not any(_inside_convex(p.vertices,point) for p in parts if p.collision)


def test_gamepad_has_free_space_between_grips_and_measured_disc_axis_landmark():
    parts,spec=asset_geometry("gamepad")
    assert spec["size"][0]==pytest.approx(.135)
    np.testing.assert_allclose(spec["reference_landmarks_local_m"]["button_disc_axis_midpoint"],[0,.022,.018175])
    point=np.array([0.,-.035,-.010])
    assert not any(_inside_convex(p.vertices,point) for p in parts if p.collision)
    grasp=spec["grasp_candidates"][0]
    rear_finger_y=grasp["center_local_m"][1]-grasp["opening_m"]/2
    sticks=[p for p in parts if p.name.startswith("stick_cap_")]
    assert rear_finger_y < min(v[1] for p in sticks for v in p.vertices)-.002


def test_gamepad_matches_rgbd_height_without_stretching_thumbsticks_or_xy():
    parts,spec=asset_geometry("gamepad")
    np.testing.assert_allclose(spec["size"],[.135,.104,.05965],atol=1e-10)
    bottom=-spec["size"][2]/2
    faces=[part for part in parts if part.name.startswith("face_plate_")]
    assert max(v[2] for part in faces for v in part.vertices)-bottom==pytest.approx(.048)
    assert spec["size"][2]-.048==pytest.approx(.01165)
    bridge=next(part for part in parts if part.name=="center_bridge")
    assert np.ptp(np.asarray(bridge.vertices)[:,1])==pytest.approx(.055)
    assert spec["grasp_candidates"][0]["surface_width_m"]==pytest.approx(.055)
    assert spec["grasp_candidates"][0]["opening_m"]==pytest.approx(.070)
    assert spec["grasp_candidates"][0]["closed_width_m"]==pytest.approx(.041)
    bounds=np.asarray(bridge.vertices)[:,2]
    contact=spec["grasp_candidates"][0]["center_local_m"][2]
    assert bounds.min()<contact<bounds.max()
    assert contact-bottom==pytest.approx(.020*.048/.0355)
    assert "not directly measured" in spec["height_evidence"]["grasp_contact_height_status"]


def test_tall_gamepad_can_be_supported_inside_bin_while_sticks_protrude_above_rim():
    from pressb.tasks.tabletop_cleanup.task import TabletopCleanupOutcome
    specs=[asset_spec(kind) for kind in KINDS]
    box=dict(center_world=[0,0,.0245],inner_min_local=[-.107,-.070,-.0205],
             inner_max_local=[.107,.070,.0245],yaw_deg=0.)
    detector=TabletopCleanupOutcome(specs,0.,box,[True,True,False])
    poses=np.array([[.0365,0,.004+specs[0]["size"][2]/2],
                    [-.0705,0,.004+specs[1]["size"][2]/2],[0,0,0]])
    q=np.tile([1.,0,0,0],(3,1));v=np.zeros((3,6))
    lifted=poses.copy();lifted[:2,2]+=.1
    detector.update(seconds=.01,positions=lifted,quaternions_wxyz=q,velocities=v,
        gripper_width_m=.036,contacts={"finger_forces_n":[[.1,.1],[.1,.1],[0,0]]})
    for seconds in [.1,.3,.7]:
        result=detector.update(seconds=seconds,positions=poses,quaternions_wxyz=q,velocities=v,
            gripper_width_m=.07,contacts={"bin_floor_forces_n":[1.,1.,0.]})
    assert result["success"]
    assert result["objects"]["gamepad"]["top_world_m"]>.049
    # Protruding in height is permitted; protruding through a side wall is not.
    poses[0,0]+=.010
    result=detector.update(seconds=.8,positions=poses,quaternions_wxyz=q,velocities=v,
        gripper_width_m=.07,contacts={"bin_floor_forces_n":[1.,1.,0.]})
    assert not result["success"] and not result["objects"]["gamepad"]["contained"]


def test_scale_scales_geometry_mass_and_grasp_and_metadata_is_not_shared_mutable():
    original=asset_spec("lidar");scaled=asset_spec("lidar",2.)
    np.testing.assert_allclose(scaled["size"],np.asarray(original["size"])*2)
    assert scaled["mass_kg"]==pytest.approx(original["mass_kg"]*8)
    original["size"][0]=77
    assert asset_spec("lidar")["size"][0]<1
    for bad in (0,-1,float("nan")):
        with pytest.raises(ValueError):asset_spec("lidar",bad)


def test_usd_local_meshes_single_rigid_root_and_static_bin_with_open_cavity():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    from pressb.tasks.tabletop_cleanup.assets import create_object,create_bin
    stage=Usd.Stage.CreateInMemory()
    quat=(math.cos(.4),0,0,math.sin(.4))
    for kind in KINDS:
        root=f"/World/{kind}"
        meta=create_object(stage,root,kind,position=(1,2,3),orientation_wxyz=quat)
        assert stage.GetPrimAtPath(root).HasAPI(UsdPhysics.RigidBodyAPI)
        transforms=UsdGeom.XformCache()
        for path in meta["collision_paths"]:
            prim=stage.GetPrimAtPath(path)
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=="convexHull"
            assert np.max(np.abs(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())))<.1
            translation=transforms.GetLocalToWorldTransform(prim).ExtractTranslation()
            np.testing.assert_allclose(translation,[1,2,3],atol=1e-8)
    bin=create_bin(stage,"/World/Bin",position=(0,0,0))
    assert not stage.GetPrimAtPath(bin["path"]).HasAPI(UsdPhysics.RigidBodyAPI)
    assert len(bin["collision_paths"])==5 and len(bin["wall_paths"])==4
    np.testing.assert_allclose(np.asarray(bin["inner_max_local"])-bin["inner_min_local"],[.214,.140,.045])
    for path in bin["collision_paths"]:
        points=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get())
        assert not _inside_convex(points,np.array([0,0,.020]))
    floor=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(bin["floor_path"])).GetPointsAttr().Get())
    assert _inside_convex(floor,np.array([0,0,-.023]))


def test_usd_rejects_invalid_cavity_and_contact_parameters():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd
    from pressb.tasks.tabletop_cleanup.assets import create_object,create_bin
    stage=Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError,match="cavity"):
        create_bin(stage,"/World/Bin",wall_thickness=.03)
    with pytest.raises(ValueError,match="dynamic_friction"):
        create_object(stage,"/World/Lidar","lidar",static_friction=.1,dynamic_friction=.5)
