"""Generic manipulation export preserves applied targets and episode boundaries."""
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import export_manipulation_dataset as export


def make_raw(root, count=7, episodes=2):
    root.mkdir(parents=True)
    manifest=dict(task_id='cube_stack', instruction='Pick up the red cube and place it on top of the blue cube.', target_episodes=episodes)
    export.write_json(root/'manifest.json', manifest)
    for episode in range(episodes):
        folder=root/'episodes'/f'episode_{episode:06d}'
        folder.mkdir(parents=True)
        state=np.zeros((count,7));state[:,0]=np.arange(count)*.01+episode*.05;state[:,6]=.06
        action=state+np.array([.02,.01,-.01,.01,-.01,0.,-.01]);action[-1]=action[-2]
        np.savez(folder/'data.npz',sim_time=np.arange(count)/30,state=state,action=action)
        export.write_json(folder/'metadata.json', dict(**{k:manifest[k] for k in ['task_id','instruction']},
            episode_index=episode, frame_count=count, fps=30, seed=episode+1, success=True,
            state_units=export.RAW_UNITS, action_units=export.RAW_UNITS, camera_resolution=[640,480],
            gripper_representation='full_width',terminal_action=export.TERMINAL_ACTION,
            initial_state_measured_rad_m=state[0].tolist(),duration_s=(count-1)/30))
        for name in export.CAMERAS.values():(folder/name).touch()
    return root


def test_conversion_preserves_full_width_and_does_not_shift_actions():
    x=np.array([[np.pi,0.,0.,0.,0.,0.,.07],[0.,np.pi/2,0.,0.,0.,0.,.05]])
    original=x.copy(); y=export.to_pap_units(x)
    np.testing.assert_allclose(y,[[180,0,0,0,0,0,7],[0,90,0,0,0,0,5]])
    np.testing.assert_array_equal(x,original)
    assert y.dtype==np.float32


@pytest.mark.parametrize('corruption',['terminal','time','units','success','language','missing_video'])
def test_corrupt_raw_episode_is_rejected(tmp_path,corruption):
    raw=make_raw(tmp_path/'raw')
    folder=raw/'episodes/episode_000000'
    meta=export.read_json(folder/'metadata.json')
    if corruption in ['terminal','time']:
        with np.load(folder/'data.npz') as z: data={k:z[k].copy() for k in z.files}
        if corruption=='terminal':data['action'][-1]=data['state'][-1]
        else:data['sim_time'][2:]+=1/30
        np.savez(folder/'data.npz',**data)
    elif corruption=='units':meta['state_units']=['deg']*6+['cm']
    elif corruption=='success':meta['success']=False
    elif corruption=='language':meta['instruction']='Other task.'
    elif corruption=='missing_video':(folder/'wrist.mp4').unlink()
    export.write_json(folder/'metadata.json',meta)
    with pytest.raises(ValueError):export.discover_episodes(raw,2)


def test_only_committed_consecutive_successes_are_discovered(tmp_path):
    raw=make_raw(tmp_path/'raw')
    (raw/'episodes/.episode_000002.inprogress').mkdir()
    (raw/'rejected/seed_20').mkdir(parents=True)
    source,entries=export.discover_episodes(raw,2)
    assert len(entries)==2
    assert source['task_id']=='cube_stack'
    meta=export.read_json(entries[1]['directory']/'metadata.json');meta['seed']=1
    export.write_json(entries[1]['directory']/'metadata.json',meta)
    with pytest.raises(ValueError,match='Repeated seed'):export.discover_episodes(raw,2)


def test_statistics_have_exact_constant_dimension_and_global_quantiles():
    stats=export.numeric_stats(np.array([[0.,.02],[2.,.02],[4.,.02]]))
    assert stats['std'][1]==0.
    assert stats['q50']==[2.,.02]
    assert stats['count']==[3]


def test_official_export_and_v21_copy_are_readable(tmp_path):
    pytest.importorskip("lerobot")
    if version("lerobot") != export.LEROBOT_VERSION:
        pytest.skip(f"Official exporter requires LeRobot {export.LEROBOT_VERSION}")
    import av
    import pyarrow.parquet as pq
    export.configure_runtime()
    raw=make_raw(tmp_path/'raw',count=9)
    for episode in range(2):
        folder=raw/'episodes'/f'episode_{episode:06d}'
        for view,name in enumerate(export.CAMERAS.values()):
            with av.open(str(folder/name),'w') as container:
                stream=container.add_stream('libx264',rate=30)
                stream.width,stream.height,stream.pix_fmt=640,480,'yuv420p'
                stream.options={'crf':'18','preset':'ultrafast','g':'2'}
                for i in range(9):
                    rgb=np.zeros((480,640,3),np.uint8)
                    rgb[:]=[20+60*episode,30+10*i,40+80*view]
                    rgb[80:150,30+i*10:130+i*10]=[200,160,100]
                    frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
    source,entries=export.discover_episodes(raw,2)
    out=tmp_path/'v3';audit=export.export_dataset(raw,out,source,entries,workers=2)
    assert audit['success'] and audit['frames']==18
    assert audit['official_reader_samples']==6
    assert set(audit['decoded_frames'].values())=={18}
    stats=export.read_json(out/'meta/stats.json')
    assert stats['action']['std'][-1]==0.
    assert stats['action']['count']==[18]
    before=deepcopy(export.read_json(out/'meta/export_manifest.json'))
    export.export_dataset(raw,out,source,entries,workers=2)
    assert before==export.read_json(out/'meta/export_manifest.json')
    legacy=tmp_path/'v21'; export.export_v21_compatibility(raw,out,legacy,entries)
    info=export.read_json(legacy/'meta/info.json')
    assert info['codebase_version']=='v2.1' and info['total_videos']==4
    for i,entry in enumerate(entries):
        path=legacy/info['data_path'].format(episode_chunk=0,episode_index=i)
        rows=pq.read_table(path).to_pylist()
        np.testing.assert_array_equal(np.array([r['action'] for r in rows],np.float32),entry['action'])
        for key,name in export.CAMERAS.items():
            copied=legacy/info['video_path'].format(episode_chunk=0,episode_index=i,video_key=key)
            assert copied.read_bytes()==(entry['directory']/name).read_bytes()
    assert len((legacy/'meta/episodes_stats.jsonl').read_text().splitlines())==2
