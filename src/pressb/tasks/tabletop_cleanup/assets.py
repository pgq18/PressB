"""Author the cleanup props as local compound rigid bodies and an open bin.

The pure CPU geometry specification is available before SimulationApp starts.
USD imports are deliberately confined to authoring functions.
"""
from __future__ import annotations

import math
import numpy as np

from .geometry import KINDS, asset_geometry, asset_spec, box


MATERIALS = {
    "plastic": ((.022,.025,.027), .52, .0),
    "housing": ((.016,.019,.021), .42, .0),
    "cover": ((.050,.055,.058), .40, .08),
    "rubber": ((.008,.011,.012), .72, .0),
    "rubber_edge": ((.023,.026,.027), .72, .0),
    "glass": ((.004,.012,.010), .15, .18),
    "metal": ((.35,.38,.38), .28, .8),
    "screw": ((.060,.065,.065), .36, .5),
    "label": ((.25,.30,.28), .7, .0),
    "label_ink": ((.05,.09,.08), .8, .0),
    "button_green": ((.12,.31,.20), .35, .0),
    "button_red": ((.37,.10,.14), .35, .0),
    "button_blue": ((.12,.25,.39), .35, .0),
    "button_pink": ((.35,.17,.32), .35, .0),
    "bin": ((.014,.016,.020), .78, .0),
    "collision": ((.0,.0,.0), 1., .0),
}


def _positive(value, name):
    value=float(value)
    if not math.isfinite(value) or value<=0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector(value,length,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(length,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return value


def _root(stage,path,position,orientation_wxyz):
    from pxr import Gf,UsdGeom
    position=_vector(position,3,"position")
    quat=_vector(orientation_wxyz,4,"orientation_wxyz")
    norm=float(np.linalg.norm(quat))
    if norm<1e-10:
        raise ValueError("orientation_wxyz must be nonzero")
    quat=quat/norm
    root=UsdGeom.Xform.Define(stage,path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddOrientOp().Set(Gf.Quatf(float(quat[0]),Gf.Vec3f(*quat[1:])))
    return root


def _mesh(stage,path,part,material):
    from pxr import Gf,UsdGeom,UsdPhysics
    from pressb.runtime.scene import _bind,PhysxSchema
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in part.vertices])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in part.faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in part.faces for i in f])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    vertices=np.asarray(part.vertices)
    mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)),Gf.Vec3f(*vertices.max(0))])
    _bind(mesh.GetPrim(),material,part.collision)
    if part.collision:
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        if PhysxSchema is not None:
            api=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            api.CreateContactOffsetAttr(.0004)
            api.CreateRestOffsetAttr(0.)
    if part.material=="collision":
        # Invisible contact counterpart for sub-fingertip-scale grille holes.
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return mesh


def _materials(stage,path,names,static_friction,dynamic_friction):
    from pressb.runtime.scene import _material
    static_friction=_positive(static_friction,"static_friction")
    dynamic_friction=_positive(dynamic_friction,"dynamic_friction")
    if dynamic_friction>static_friction:
        raise ValueError("dynamic_friction must not exceed static_friction")
    return {name:_material(stage,f"{path}/Looks/{name}",MATERIALS[name][0],
             roughness=MATERIALS[name][1],metallic=MATERIALS[name][2],
             static_friction=static_friction,dynamic_friction=dynamic_friction)
            for name in sorted(set(names))}


def create_object(stage,path,kind,*,position=(0.,0.,0.),orientation_wxyz=(1.,0.,0.,0.),
                  mass_kg=None,static_friction=.8,dynamic_friction=.65,scale=1.):
    """Return geometry metadata; only root Xform is a dynamic rigid body."""
    from pxr import Gf,UsdPhysics
    from pressb.runtime.scene import PhysxSchema
    parts,_=asset_geometry(kind,float(scale))
    spec=asset_spec(kind,float(scale))
    mass_kg=_positive(spec["mass_kg"] if mass_kg is None else mass_kg,"mass_kg")
    root=_root(stage,path,position,orientation_wxyz)
    mats=_materials(stage,path,[p.material for p in parts],static_friction,dynamic_friction)
    collision_paths=[]
    for part in parts:
        mesh=_mesh(stage,f"{path}/Geometry/{part.name.replace('-', 'm')}",part,mats[part.material])
        if part.collision:
            collision_paths.append(str(mesh.GetPath()))
    body=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    mass=UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(mass_kg)
    # Explicit total mass; PhysX computes COM/inertia from the authored convex
    # compound instead of treating the concave gamepad as a bounding box.
    if PhysxSchema is not None:
        api=PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        api.CreateEnableCCDAttr(True)
        api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        api.CreateSleepThresholdAttr(0.)
        PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.)
    return dict(spec,path=str(root.GetPath()),collision_paths=collision_paths,mass_kg=mass_kg,
                center_world=list(map(float,position)),dynamic=True,
                static_friction=float(static_friction),dynamic_friction=float(dynamic_friction))


def create_bin(stage,path,*,size_xyz=(.222,.148,.049),wall_thickness=.004,
               position=(0.,0.,0.),orientation_wxyz=(1.,0.,0.,0.),
               static_friction=.8,dynamic_friction=.65):
    """Static open cavity: bottom and four walls, no hidden roof/hull collider."""
    size=_vector(size_xyz,3,"size_xyz")
    if np.any(size<=0):
        raise ValueError("bin size_xyz must be positive")
    thickness=_positive(wall_thickness,"wall_thickness")
    if np.any(size<=2*thickness):
        raise ValueError("bin wall thickness must leave a nonempty cavity")
    root=_root(stage,path,position,orientation_wxyz)
    material=_materials(stage,path,["bin"],static_friction,dynamic_friction)["bin"]
    x,y,z=size
    parts=[box("floor",(x,y,thickness),(0,0,-z/2+thickness/2),"bin")]
    for sign in (-1,1):
        parts.append(box(f"wall_x_{sign}",(thickness,y,z-thickness),
                         (sign*(x-thickness)/2,0,thickness/2),"bin"))
        parts.append(box(f"wall_y_{sign}",(x-2*thickness,thickness,z-thickness),
                         (0,sign*(y-thickness)/2,thickness/2),"bin"))
    paths=[str(_mesh(stage,f"{path}/Geometry/{p.name.replace('-', 'm')}",p,material).GetPath()) for p in parts]
    return dict(path=str(root.GetPath()),size=size.tolist(),size_xyz=size.tolist(),
        wall_thickness=thickness,center_world=list(map(float,position)),dynamic=False,
        collision_paths=paths,floor_path=paths[0],wall_paths=paths[1:],
        inner_min_local=[-x/2+thickness,-y/2+thickness,-z/2+thickness],
        inner_max_local=[x/2-thickness,y/2-thickness,z/2],
        support_vertices=[list(p) for part in parts for p in part.vertices])
