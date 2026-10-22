#!/usr/bin/env python3
"""Map and visualize visible, collision-screened tabletop press positions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/piper_table_buttons.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--step', type=float, default=.01)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    from pressb.kinematics import PiperKinematics
    from pressb.table_buttons.placement import ReachablePlacement
    placement = ReachablePlacement(PiperKinematics(ROOT/cfg['robot_urdf'], tip_offset=.1358), cfg)
    start = time.monotonic()
    grid = placement.grid(args.step)
    args.output.mkdir(parents=True, exist_ok=False)
    pts, visible, feasible = (grid[key] for key in ('points_xy', 'visible_mask', 'reachable_mask'))
    pitch = np.array([np.nan if a is None else a['pitch_deg'] for a in grid['approaches']])
    hover = np.array([np.nan if a is None else a['hover_z_m'] for a in grid['approaches']])
    np.savez_compressed(args.output/'workspace_map.npz', points_xy=pts, visible=visible,
                        feasible=feasible, pitch_deg=pitch, hover_z_m=hover, step_m=args.step)
    # Normalize the grid acceptance fraction by the analytically clipped
    # visible area, so edge cells cannot overstate the camera's footprint.
    area = grid['domain']['area_m2']*int(feasible.sum())/max(1, int(visible.sum()))
    bounds = np.column_stack([pts[feasible].min(axis=0), pts[feasible].max(axis=0)])
    report = dict(grid_step_m=args.step, visible_grid_points=int(visible.sum()),
                  feasible_grid_points=int(feasible.sum()), approximate_area_m2=area,
                  area_estimator='exact visible polygon area times feasible fraction of visible grid points',
                  raw_grid_cell_area_m2=int(feasible.sum())*args.step**2,
                  old_area_m2=.09*.28, approximate_area_ratio=area/(.09*.28),
                  feasible_grid_bounds_xy_m=bounds.tolist(), reasons=dict(Counter(grid['reasons'])),
                  camera_domain=grid['domain'], collision_model=placement.clearance.model_info,
                  pitch_counts=dict(Counter(str(v) for v in pitch[feasible])),
                  wall_time_s=time.monotonic()-start,
                  note='Area is a grid estimate; actual sampling re-solves exact positions and checks all three button paths.')
    # Keep boundary poses for separate Isaac runs rather than presenting the
    # grid as a physical success-rate evaluation.
    candidates = pts[feasible]
    selections = {'near': np.argmin(candidates[:, 0]), 'far': np.argmax(candidates[:, 0]),
                  'left': np.argmax(candidates[:, 1]), 'right': np.argmin(candidates[:, 1]),
                  'far_left': np.argmax(candidates[:, 0]+candidates[:, 1]),
                  'far_right': np.argmax(candidates[:, 0]-candidates[:, 1])}
    report['boundary_candidates_xy_m'] = {name: candidates[index].tolist() for name, index in selections.items()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    draw(cfg, pts, visible, feasible, args.step, report, args.image, args.output)


def draw(cfg, pts, visible, feasible, step, report, image_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties, fontManager
    from matplotlib.patches import Polygon, Patch
    from PIL import Image
    from pressb.table_buttons.task import project_base_points

    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font).exists():
        fontManager.addfont(font)
        plt.rcParams['font.family'] = FontProperties(fname=font).get_name()
    plt.rcParams.update({'font.size': 12, 'axes.unicode_minus': False})
    corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*step/2
    cells = pts[:, None, :]+corners[None]
    flat = cells.reshape(-1,2)
    pixels, _ = project_base_points(np.c_[flat,np.zeros(len(flat))], cfg['up_camera'])
    image_cells = pixels.reshape(-1,4,2)
    fig = plt.figure(figsize=(14,7.5), facecolor='#F5F7FA')
    fig.text(.04,.95,'扩展后的按钮随机生成区域', fontsize=23, weight='bold', color='#152B42')
    fig.text(.04,.90,f"可行区域约 {report['approximate_area_m2']*1e4:.0f} cm² · 原来的 {report['approximate_area_ratio']:.1f} 倍 · {step*100:g} cm 网格估计", fontsize=17, color='#35516B')
    ax = fig.add_axes([.04,.205,.54,.62])
    ax.imshow(Image.open(image_path))
    ax.add_collection(PolyCollection(image_cells[visible & ~feasible], facecolor='#EC8554', alpha=.35, edgecolor='none'))
    ax.add_collection(PolyCollection(image_cells[feasible], facecolor='#54E5B0', alpha=.28, edgecolor='none'))
    old = np.array([[.30,-.14],[.39,-.14],[.39,.14],[.30,.14]])
    old_pixels,_ = project_base_points(np.c_[old,np.zeros(4)],cfg['up_camera'])
    ax.add_patch(Polygon(old_pixels, fill=False, edgecolor='#FFE56A', lw=2.2, linestyle='--'))
    ax.set_title('第三视角 · 采样范围在桌面上的投影',loc='left',pad=12,fontsize=14,color='#152B42')
    ax.axis('off')
    bx = fig.add_axes([.65,.205,.30,.62])
    bx.set_facecolor('white')
    top_cells = cells[..., [1,0]]*100
    bx.add_collection(PolyCollection(top_cells[visible & ~feasible], facecolor='#F4B392', edgecolor='none'))
    bx.add_collection(PolyCollection(top_cells[feasible], facecolor='#96DDBE', edgecolor='none'))
    bx.add_patch(Polygon(old[:,[1,0]]*100,fill=False,edgecolor='#B88400',lw=1.7,linestyle='--'))
    camera = np.array(report['camera_domain']['polygon_xy_m'])
    bx.add_patch(Polygon(camera[:,[1,0]]*100,fill=False,edgecolor='#6B7E8D',lw=1.2))
    bx.plot(0,0,'o',color='#203D53',markersize=8)
    bx.text(0,2,'机械臂底座',ha='center',fontsize=11,color='#203D53')
    bx.set_xlim(34,-34); bx.set_ylim(-3,56); bx.set_aspect('equal')
    bx.set_xlabel('Y / cm（正向为机械臂左侧）'); bx.set_ylabel('X / cm（向前）')
    bx.set_title('俯视 · 相机可见区域与可执行按压区域的交集',loc='left',pad=12,fontsize=13,color='#152B42')
    bx.grid(alpha=.15)
    for spine in bx.spines.values(): spine.set_color('#BBC7D1')
    fig.legend(handles=[Patch(facecolor='#96DDBE',label='通过可见性、IK和离散路径碰撞检查'),
                        Patch(facecolor='#F4B392',label='可见但未通过按压路径检查'),
                        Patch(facecolor='none',edgecolor='#B88400',linestyle='--',label='原 9 × 28 cm 范围')],
               loc='lower left',bbox_to_anchor=(.04,.11),frameon=False,ncol=2,fontsize=11)
    fig.text(.04,.055,'每次采样重新检查实际位置和整组三按钮路径；边界会随姿态、按钮尺寸和相机参数变化。',fontsize=12,color='#526C80')
    fig.savefig(output/'expanded_sampling_range.png',dpi=160,facecolor=fig.get_facecolor())
    fig.savefig(output/'expanded_sampling_range.pdf',facecolor=fig.get_facecolor())


if __name__ == '__main__':
    main()
