"""Measured tabletop props as local, closed compound meshes (metres).

No USD/Isaac dependency: reset sampling and containment use the same vertices
as authoring. Dimensions are estimates from the supplied multi-angle videos;
small unmeasured features are geometric approximations, not scanned meshes.
X is the gamepad's left/right axis, +Y its shoulder-button side, +Z up.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from functools import lru_cache

import numpy as np


# The user confirmed 54 x 39 mm describes the speaker's main enclosure.
# Ear offsets/radii remain separate visual estimates from the supplied video.
SPEAKER_BODY_SIZE = (.054, .039, .023)
SPEAKER_EAR_CENTER_XY = (.0295, .0205)
SPEAKER_EAR_OUTER_RADIUS = .004
SPEAKER_EAR_HOLE_RADIUS = .0018


@dataclass(frozen=True)
class Part:
    name: str
    vertices: tuple
    faces: tuple
    material: str = "plastic"
    collision: bool = True


def _part(name, vertices, faces, material="plastic", collision=True):
    vertices = np.asarray(vertices, dtype=float)
    faces = [tuple(face) for face in faces]
    # Consistent outward orientation also makes CPU signed-volume QA useful.
    volume = sum(np.dot(vertices[f[0]], np.cross(vertices[f[i]], vertices[f[i+1]]))
                 for f in faces for i in range(1, len(f)-1))/6
    if volume < 0:
        faces = [face[::-1] for face in faces]
    return Part(name, tuple(map(tuple, vertices)), tuple(faces), material, collision)


def _loft(name, rings, material="plastic", collision=True):
    n = len(rings[0])
    if n < 3 or any(len(r) != n for r in rings):
        raise ValueError("loft requires equal rings of at least three vertices")
    vertices = [p for ring in rings for p in ring]
    faces = [tuple(reversed(range(n))), tuple(range((len(rings)-1)*n, len(rings)*n))]
    for k in range(len(rings)-1):
        for j in range(n):
            faces.append((k*n+j, k*n+(j+1)%n, (k+1)*n+(j+1)%n, (k+1)*n+j))
    return _part(name, vertices, faces, material, collision)


def rounded_box(name, size, center=(0., 0., 0.), radius=.002, material="plastic", collision=True):
    x, y, z = np.asarray(size, float)/2
    r = min(float(radius), x, y)
    outline = []
    for cx, cy, start in ((x-r,y-r,0),(-x+r,y-r,90),(-x+r,-y+r,180),(x-r,-y+r,270)):
        for angle in np.linspace(start, start+90, 5):
            a = math.radians(float(angle))
            outline.append((cx+r*math.cos(a), cy+r*math.sin(a)))
    rings = [[(px+center[0], py+center[1], pz+center[2]) for px, py in outline] for pz in (-z,z)]
    return _loft(name, rings, material, collision)


def box(name, size, center=(0., 0., 0.), material="plastic", collision=True):
    x, y, z = np.asarray(size, float)/2
    outline = [(x,y),(-x,y),(-x,-y),(x,-y)]
    return _loft(name, [[(px+center[0],py+center[1],pz+center[2]) for px,py in outline]
                        for pz in (-z,z)], material, collision)


def cylinder(name, radius, height, center, material="plastic", collision=True, segments=32, bevel=0.):
    r, h = float(radius), float(height)/2
    profile = [(r,-h),(r,h)] if not bevel else [(r-bevel,-h),(r,-h+bevel),(r,h-bevel),(r-bevel,h)]
    rings = [[(center[0]+rr*math.cos(a),center[1]+rr*math.sin(a),center[2]+z)
              for a in np.linspace(0,2*math.pi,segments,endpoint=False)] for rr,z in profile]
    return _loft(name,rings,material,collision)


def annulus(name, outer, inner, height, center, material="plastic", collision=True, segments=16):
    """Convex quadrilateral sectors, leaving the real through-hole unfilled."""
    parts=[]
    for j in range(segments):
        a,b = 2*math.pi*j/segments,2*math.pi*(j+1)/segments
        xy=[(outer*math.cos(a),outer*math.sin(a)),(outer*math.cos(b),outer*math.sin(b)),
            (inner*math.cos(b),inner*math.sin(b)),(inner*math.cos(a),inner*math.sin(a))]
        rings=[[(x+center[0],y+center[1],z+center[2]) for x,y in xy] for z in (-height/2,height/2)]
        parts.append(_loft(f"{name}_{j:02d}",rings,material,collision))
    return parts


def _square_hole_tile(name, half, radius, thickness, center, material="cover"):
    # Each small tile is square outside and has a real circular hole inside.
    # Joining the visual sectors avoids a texture pretending to be a hole.
    parts=[]
    for j in range(16):
        a,b = 2*math.pi*j/16,2*math.pi*(j+1)/16
        points=[]
        for angle,inner in ((a,False),(b,False),(b,True),(a,True)):
            co,si=math.cos(angle),math.sin(angle)
            r=radius if inner else half/max(abs(co),abs(si))
            points.append((center[0]+r*co,center[1]+r*si))
        rings=[[(x,y,center[2]+z) for x,y in points] for z in (-thickness/2,thickness/2)]
        parts.append(_loft(f"{name}_{j}",rings,material,False))
    return parts


def _merge(name, parts, material, collision=False):
    vertices,faces=[],[]
    for part in parts:
        offset=len(vertices);vertices.extend(part.vertices)
        faces.extend(tuple(v+offset for v in f) for f in part.faces)
    return _part(name,vertices,faces,material,collision)


def _speaker():
    p=[]
    bx,by,bz=SPEAKER_BODY_SIZE
    hx,hy=bx/2,by/2
    p.append(rounded_box("housing",(bx,by,bz-.003),(0,0,(bz-.003)/2),radius=.002,material="housing"))
    # Top is a shallow hollow cap: the dark diaphragm lies visibly below holes.
    p.append(box("diaphragm",(.030,.030,.001),(0,0,.0200),"rubber",False))
    pitch=.0024;tilehalf=pitch/2;n=9;gridhalf=n*pitch/2
    tiles=[]
    for ix in range(n):
        for iy in range(n):
            x,y=(ix-(n-1)/2)*pitch,(iy-(n-1)/2)*pitch
            if x*x+y*y < .0099**2:
                tiles.extend(_square_hole_tile(f"grill_{ix}_{iy}",tilehalf,.00078,.0012,(x,y,.0224)))
            else:
                tiles.append(box(f"tile_{ix}_{iy}",(pitch,pitch,.0012),(x,y,.0224),"cover",False))
    p.append(_merge("perforated_grille",tiles,"cover"))
    for side in (-1,1):
        p.append(box(f"lid_x_{side}",(hx-gridhalf,by,.0012),
                     (side*(hx+gridhalf)/2,0,.0224),"cover"))
        p.append(box(f"lid_y_{side}",(2*gridhalf,hy-gridhalf,.0012),
                     (0,side*(hy+gridhalf)/2,.0224),"cover"))
    # A collision-only 1.2mm grille slab approximates <1.6mm holes: PiPER
    # fingertips cannot enter those perforations. Ear holes remain physical.
    p.append(box("grille_contact",(2*gridhalf,2*gridhalf,.0012),(0,0,.0224),"collision"))
    for sx in (-1,1):
        for sy in (-1,1):
            center=(sx*SPEAKER_EAR_CENTER_XY[0],sy*SPEAKER_EAR_CENTER_XY[1],.0015)
            p.extend(annulus(f"mount_ear_{sx}_{sy}",SPEAKER_EAR_OUTER_RADIUS,SPEAKER_EAR_HOLE_RADIUS,.003,center,"housing",True,12))
            p.append(cylinder(f"lid_screw_{sx}_{sy}",.0019,.00025,(sx*(hx-.004),sy*(hy-.004),.0231),"screw",False,20))
            p.append(box(f"screw_slot_{sx}_{sy}",(.0028,.00035,.0001),(sx*(hx-.004),sy*(hy-.004),.02327),"rubber",False))
    for sign in (-1,1):
        # Side grille recesses and four curved-cover vent grooves from video.
        for row in range(4):
            p.append(box(f"side_slot_{sign}_{row}",(.014,.00018,.00075),(0,sign*(hy+.00006),.006+row*.0024),"rubber",False))
            p.append(rounded_box(f"top_vent_{sign}_{row}",(.0007,.0055,.00013),
                (sign*(.014+row*.0015),0,.02307),radius=.0003,material="rubber",collision=False))
    return p, .045, [dict(center=(0,0,.010),axis=(0,1,0),width=by,opening=.052,closed=by-.003),
                       dict(center=(0,0,.010),axis=(1,0,0),width=bx,opening=.067,closed=bx-.003)]


def _lidar():
    p=[rounded_box("square_base",(.038,.038,.021),(0,0,.0105),radius=.003,material="housing"),
       cylinder("scanner_drum",.019,.013,(0,0,.0275),"glass",True,48,bevel=.0005),
       cylinder("top_cover",.0185,.0003,(0,0,.03385),"glass",False,48),
       cylinder("silver_rim",.01905,.00055,(0,0,.0315),"metal",False,48)]
    # Fine external silver rim is visual; the drum supplies matching contact.
    for sx in (-1,1):
        for sy in (-1,1):
            p.extend(annulus(f"corner_boss_{sx}_{sy}",.002,.0011,.0007,
                             (sx*.0155,sy*.0155,.0211),"metal",False,12))
    p.append(box("label",(.015,.0001,.007),(0,-.01905,.010),"label",False))
    for i in range(5):
        p.append(box(f"label_line_{i}",(.012,.00012,.0003),(0,-.01912,.0075+i*.001),"label_ink",False))
    return p,.055,[dict(center=(0,0,.010),axis=(0,1,0),width=.038,opening=.050,closed=.035),
                    dict(center=(0,0,.010),axis=(1,0,0),width=.038,opening=.050,closed=.035)]


# Fresh tabletop RGB-D on 2026-11-09: face-disc surfaces are about 47–49 mm
# above the table, thumbstick tops 57–60 mm. XY dimensions remain photo priors.
# This corrects the old thin-shell estimate; it is not a scanned CAD model.
GAMEPAD_FACE_SURFACE_HEIGHT_M = .048
GAMEPAD_BRIDGE_WIDTH_M = .055
_GAMEPAD_PRIOR_FACE_SURFACE_HEIGHT_M = .0355


def _gamepad_height(z):
    """Thicken the shell, retaining raised controls' height above the face."""
    z = np.asarray(z, dtype=float)
    ratio = GAMEPAD_FACE_SURFACE_HEIGHT_M/_GAMEPAD_PRIOR_FACE_SURFACE_HEIGHT_M
    return np.where(z <= _GAMEPAD_PRIOR_FACE_SURFACE_HEIGHT_M, z*ratio,
                    z+GAMEPAD_FACE_SURFACE_HEIGHT_M-_GAMEPAD_PRIOR_FACE_SURFACE_HEIGHT_M)


def _gamepad():
    p=[]
    p.append(rounded_box("center_bridge",(.088,GAMEPAD_BRIDGE_WIDTH_M,.023),(0,.009,.021),radius=.008,material="housing"))
    for sign in (-1,1):
        cx=sign*.043
        p.append(cylinder(f"face_disc_{sign}",.0245,.021,(cx,.014,.0235),"housing",True,40,bevel=.002))
        p.append(cylinder(f"face_plate_{sign}",.0235,.0015,(cx,.014,.03475),"cover",True,40))
        # Tapered grips sweep rearward; each is its own convex body, preserving
        # the large free space between them instead of a hull of the gamepad.
        rings=[]
        for yy,xx,rx,zc,rz in ((.014,.043,.020,.019,.016),(-.010,.047,.0205,.018,.018),
                                (-.038,.049,.017,.015,.015),(-.054,.049,.010,.012,.011)):
            rings.append([(sign*xx+rx*math.cos(a),yy,zc+rz*math.sin(a))
                          for a in np.linspace(0,2*math.pi,24,endpoint=False)])
        # Closed tapered end gives a rounded silhouette without a square grip.
        rings.append([(sign*.049+.003*math.cos(a),-.060,.011+.004*math.sin(a))
                      for a in np.linspace(0,2*math.pi,24,endpoint=False)])
        p.append(_loft(f"grip_{sign}",rings,"housing"))
        p.append(rounded_box(f"shoulder_{sign}",(.023,.010,.011),(cx,.039,.026),radius=.004,material="rubber"))
        p.append(cylinder(f"stick_socket_{sign}",.013,.004,(sign*.022,-.012,.034),"cover",True,32))
        p.append(cylinder(f"stick_stem_{sign}",.005,.006,(sign*.022,-.012,.039),"rubber",True,24))
        p.append(cylinder(f"stick_cap_{sign}",.0105,.006,(sign*.022,-.012,.044),"rubber",True,32,bevel=.0012))
        p.extend(annulus(f"stick_cap_rim_{sign}",.0092,.0087,.0002,(sign*.022,-.012,.04705),"rubber_edge",False,32))
    # Four separate direction keys give the real segmented cross outline.
    for j,(dx,dy) in enumerate(((0,.0078),(.0078,0),(0,-.0078),(-.0078,0))):
        p.append(rounded_box(f"dpad_{j}",(.0075,.0075,.003),(-.043+dx,.014+dy,.037),radius=.0012,material="rubber"))
        p.append(rounded_box(f"dpad_inset_{j}",(.0045,.0045,.0002),(-.043+dx,.014+dy,.0386),radius=.0006,material="cover",collision=False))
    for j,(dx,dy,material) in enumerate(((0,.008,"button_green"),(.008,0,"button_red"),
                                       (0,-.008,"button_blue"),(-.008,0,"button_pink"))):
        p.append(cylinder(f"face_button_{j}",.0045,.0035,(.043+dx,.014+dy,.03725),"rubber",True,24,bevel=.0005))
        p.append(cylinder(f"face_button_color_{j}",.0029,.00025,(.043+dx,.014+dy,.0391),material,False,24))
    for j,(x,y,w,h) in enumerate(((-.013,.012,.006,.004),(.013,.012,.006,.004),(0,-.002,.007,.004))):
        p.append(rounded_box(f"function_button_{j}",(w,h,.0025),(x,y,.034),radius=.0008,material="rubber"))
    # Keep the separate grip cavities and all XY points. A continuous Z map
    # stretches the shell below the face plate; joystick stems/caps above it
    # translate upward without becoming disproportionately tall themselves.
    corrected = []
    for part in p:
        vertices = np.asarray(part.vertices).copy()
        vertices[:,2] = _gamepad_height(vertices[:,2])
        corrected.append(replace(part,vertices=tuple(map(tuple,vertices))))
    # The bridge's vertical contact band is inferred from this shell geometry,
    # not a directly measured hidden sidewall/contact patch.
    return corrected,.180,[dict(center=(0,.009,float(_gamepad_height(.020))),axis=(0,1,0),
        width=GAMEPAD_BRIDGE_WIDTH_M,opening=.090,closed=.041)]


KINDS=("gamepad","speaker","lidar")


@lru_cache(maxsize=12)
def asset_geometry(kind, scale=1.):
    if kind not in KINDS:
        raise ValueError(f"Unknown cleanup object {kind!r}; expected {KINDS}")
    scale=float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Asset scale must be finite and positive")
    parts,mass,grasps={"gamepad":_gamepad,"speaker":_speaker,"lidar":_lidar}[kind]()
    all_vertices=np.concatenate([np.asarray(p.vertices) for p in parts if p.material!="collision"])
    lower,upper=all_vertices.min(0),all_vertices.max(0)
    origin=(lower+upper)/2
    parts=tuple(replace(p,vertices=tuple(map(tuple,(np.asarray(p.vertices)-origin)*scale))) for p in parts)
    support=np.concatenate([np.asarray(p.vertices) for p in parts if p.material!="collision"])
    collision=np.concatenate([np.asarray(p.vertices) for p in parts if p.collision])
    # An actual outer support hull, not AABB corners. Redundant interior mesh
    # vertices cannot change containment in a rectangular box and would make
    # per-tick 64-environment outcome checks unnecessarily expensive. This hull
    # is metadata only: collision remains separate pieces with open cavities.
    from scipy.spatial import ConvexHull
    def support_hull(vertices):
        unique=np.unique(np.round(vertices,12),axis=0)
        return unique[ConvexHull(unique).vertices].tolist()
    spec=dict(kind=kind,size=((upper-lower)*scale).tolist(),mass_kg=mass*scale**3,
        support_vertices=support_hull(support),
        hull_vertices=support_hull(collision),
        grasp_candidates=[dict(center_local_m=((np.asarray(g["center"])-origin)*scale).tolist(),
            closing_axis_local=list(g["axis"]),surface_width_m=g["width"]*scale,
            opening_m=g["opening"]*scale,closed_width_m=g["closed"]*scale,
            approach_axis_local=[0,0,-1]) for g in grasps],
        modeling_notes="Dimensions approximate supplied measurements; local origin is complete visual AABB centre.",
        collision_approximation="Separate convex pieces; speaker grille holes below 1.6 mm have a solid collision cover, ear holes remain open.",
        reference_landmarks_local_m=({"button_disc_axis_midpoint": ((np.array([0.,.014,GAMEPAD_FACE_SURFACE_HEIGHT_M])-origin)*scale).tolist()} if kind=="gamepad" else {}),
        height_evidence=({"source":"2026-11-09 fresh global/wrist RGB-D tabletop-relative depth",
            "face_surface_height_m":GAMEPAD_FACE_SURFACE_HEIGHT_M*scale,
            "observed_face_range_m":[.047*scale,.049*scale],
            "observed_thumbstick_top_range_m":[.057*scale,.060*scale],
            "observed_bridge_surface_medians_m":[.04267*scale,.04481*scale],
            "bridge_depth_scope":"Visible top only in global/wrist depth; lower surface and wall thickness not measured",
            "bridge_width_m":GAMEPAD_BRIDGE_WIDTH_M*scale,
            "bridge_width_estimated_range_m":[.050*scale,.060*scale],
            "bridge_width_status":"Approximate projected central sidewall envelope with partial joystick occlusion; not a precise measured cross section",
            "shape_update":"Shell Z scaled below the disc surface; raised controls retain their offset above it; overall XY envelope unchanged, central bridge Y width separately refined",
            "grasp_contact_height_status":"Central bridge contact band inferred from corrected shell; not directly measured"}
            if kind=="gamepad" else None),
        measured_body_size=([v*scale for v in SPEAKER_BODY_SIZE] if kind=="speaker" else None),
        mounting_ears=({"center_xy_abs_m":[v*scale for v in SPEAKER_EAR_CENTER_XY],
                        "outer_radius_m":SPEAKER_EAR_OUTER_RADIUS*scale,
                        "hole_radius_m":SPEAKER_EAR_HOLE_RADIUS*scale} if kind=="speaker" else None),
        num_visual_parts=sum(p.material!="collision" for p in parts),
        num_collision_parts=sum(p.collision for p in parts))
    return parts,spec


def asset_spec(kind, scale=1.):
    """Independent mutable metadata; CPU callers never import USD/Isaac."""
    from copy import deepcopy
    return deepcopy(asset_geometry(kind,scale)[1])
