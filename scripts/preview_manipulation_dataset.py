#!/usr/bin/env python3
"""Summarize accepted random layouts and render original-rate example videos."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    episodes=[]
    for folder in sorted((args.raw/'episodes').glob('episode_*')):
        meta=json.loads((folder/'metadata.json').read_text())
        if meta['success'] is not True:raise ValueError('Rejected episode in accepted dataset')
        episodes.append((folder,meta))
    manifest=json.loads((args.raw/'manifest.json').read_text())
    if len(episodes)!=manifest['target_episodes']:raise ValueError('Dataset not complete')
    selected=[];seen=set();ranges={};counts=Counter()
    for folder,meta in episodes:
        objects=meta['reset_spec']['objects'];key=tuple(sorted(objects));counts['+'.join(key)]+=1
        if key not in seen:selected.append((folder,meta));seen.add(key)
        for name,pose in objects.items():
            values=[*pose['center_xy'],pose['yaw_deg']]
            r=ranges.setdefault(name,{'min':values.copy(),'max':values.copy(),'count':0})
            r['count']+=1;r['min']=[min(a,b) for a,b in zip(r['min'],values)];r['max']=[max(a,b) for a,b in zip(r['max'],values)]
    if len(selected)==1:
        selected=[episodes[i] for i in sorted(set([0,len(episodes)//2,len(episodes)-1]))]
    selected=selected[:3]
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    clips=[]
    for folder,meta in selected:
        clip=args.output/f"episode_{meta['episode_index']:06d}_up_wrist.mp4"
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(folder/'up.mp4'),'-i',str(folder/'wrist.mp4'),
                 '-filter_complex_threads','2','-filter_complex','[0:v][1:v]vstack=inputs=2[out]','-map','[out]','-an','-c:v','libx264','-threads','2','-crf','19',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(clip)]
        subprocess.run(command,check=True);clips.append(clip)
    longest=max(meta['frame_count']/meta['fps'] for _,meta in selected)
    command=['ffmpeg','-hide_banner','-loglevel','error','-y'];filters=[]
    for i,(clip,(_,meta)) in enumerate(zip(clips,selected)):
        command+=['-i',str(clip)]
        label=f"Episode {meta['episode_index']} - "+' + '.join(sorted(meta['reset_spec']['objects']))
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={longest},trim=duration={longest},pad=iw:ih+50:0:50:color=0x17232d,drawtext=fontfile={font}:text={label}:fontsize=20:fontcolor=white:x=(w-tw)/2:y=14[v{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(len(clips)))+f'hstack=inputs={len(clips)}[out]')
    command+=['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','2','-crf','19',
              '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'examples_1x.mp4')]
    subprocess.run(command,check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','1','-i',str(args.output/'examples_1x.mp4'),
                    '-frames:v','1','-update','1',str(args.output/'examples.jpg')],check=True)
    summary=dict(task_id=manifest['task_id'],episodes=len(episodes),pair_counts=dict(counts),observed_initial_ranges=ranges,
        range_columns=['base_X_m','base_Y_m','yaw_deg'],examples=[m['episode_index'] for _,m in selected],
        total_frames=sum(m['frame_count'] for _,m in episodes),video_rate='Original 30 Hz, shorter examples hold final frame; no speedup',
        success_selection='Only physically accepted episodes; rejection records retained separately',noise=manifest['noise_deg'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
