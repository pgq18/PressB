"""Generic joint7 training accepts new tasks without weakening old contracts."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pressb.model_sources import activate_source
activate_source('vlajepa')
from pressb.training.vlajepa_joint7_data import Joint7LeRobotDataset, SEMANTICS, CAMERA_KEYS
from pressb.training.image_transforms import build_image_transforms


@pytest.fixture
def dataset_root(tmp_path):
    root=tmp_path/'dataset'
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    instruction='Put every object on the table into the box.'
    info=dict(codebase_version='v3.0',fps=30,total_episodes=10,total_tasks=1,total_frames=50,
              video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
              features={k:dict(dtype='float32',shape=[7]) for k in ('observation.state','action')})
    for key in CAMERA_KEYS:
        path=root/info['video_path'].format(video_key=key,chunk_index=0,file_index=0)
        path.parent.mkdir(parents=True);path.touch()
    entries,metadata,numeric=[],[],[]
    for ep in range(10):
        raw=np.array([ep*.01,0,0,0,0,0,.06])
        state=raw*np.array([180/np.pi]*6+[100.])
        entries.append(dict(episode_index=ep,frame_count=5,seed=ep+1,success=True,instruction=instruction,
                            initial_state_measured_rad_m=raw.tolist()))
        meta=dict(episode_index=ep,length=5,tasks=[instruction],dataset_from_index=ep*5,dataset_to_index=ep*5+5)
        for key in CAMERA_KEYS:
            meta.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':0,
                         f'videos/{key}/from_timestamp':ep*5/30,f'videos/{key}/to_timestamp':(ep*5+5)/30})
        metadata.append(meta)
        for frame in range(5):
            command=state.copy();command[6]=6-min(frame+1,4)
            numeric.append(dict(index=ep*5+frame,episode_index=ep,frame_index=frame,task_index=0,
                timestamp=frame/30,**{'observation.state':state.tolist(),'action':command.tolist()}))
    manifest=dict(schema_version=1,task_id='new_custom_task',instruction=instruction,
                  episodes=entries,state_action_semantics=deepcopy(SEMANTICS))
    (root/'meta/info.json').write_text(json.dumps(info))
    (root/'meta/export_manifest.json').write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist(metadata),root/'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(numeric),root/'data/chunk-000/file-000.parquet')
    return root


def test_arbitrary_task_episode_split_and_fixed_units(dataset_root):
    train=Joint7LeRobotDataset(dataset_root,split='train',sample_stride=2)
    val=Joint7LeRobotDataset(dataset_root,split='val')
    assert len(train.source_episode_ids)==9 and len(val.source_episode_ids)==1
    assert set(train.source_episode_ids).isdisjoint(val.source_episode_ids)
    assert sorted(train.source_episode_ids+val.source_episode_ids)==list(range(10))
    assert train.group=='new_custom_task' and len(train)==27
    np.testing.assert_allclose(train._states[:,6],.75)
    np.testing.assert_allclose(train._actions[:5,6],np.array([5,4,3,2,2])/8)
    np.testing.assert_allclose(train._states[5,0],.01/np.pi,rtol=1e-6)


def test_boundary_padding_camera_order_and_no_validation_augmentation(dataset_root,monkeypatch):
    val=Joint7LeRobotDataset(dataset_root,split='val',image_transforms=lambda _: (_ for _ in ()).throw(AssertionError()))
    assert val.image_transforms is None
    def decode(self,path,ids):
        return np.stack([np.full((4,4,3),20 if '.up/' in path else 200,dtype=np.uint8) for _ in ids])
    monkeypatch.setattr(Joint7LeRobotDataset,'_video_frames',decode)
    e=val.source_episode_ids[0]; item=val.get_frame(e,4)
    assert item['state'].shape==(1,7) and item['action'].shape==(7,7)
    assert item['video'].shape==(2,8,4,4,3)
    assert item['video'][0,0,0,0,0]==20 and item['video'][1,0,0,0,0]==200
    np.testing.assert_array_equal(item['action_is_pad'],[False]+[True]*6)
    np.testing.assert_array_equal(item['action'],np.tile(item['action'][0],(7,1)))
    assert item['lang']=='Put every object on the table into the box.'


@pytest.mark.parametrize('mutation',['units','unsuccessful','duplicate_seed','timing'])
def test_ambiguous_export_contract_rejected(dataset_root,mutation):
    path=dataset_root/'meta/export_manifest.json';m=json.loads(path.read_text())
    if mutation=='units':m['state_action_semantics']['state_units']=['rad']*6+['m']
    elif mutation=='unsuccessful':m['episodes'][0]['success']=False
    elif mutation=='duplicate_seed':m['episodes'][1]['seed']=m['episodes'][0]['seed']
    else:m['state_action_semantics']['action']='previous_interval_target'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):Joint7LeRobotDataset(dataset_root)


def test_numeric_terminal_alignment_is_validated(dataset_root):
    path=dataset_root/'data/chunk-000/file-000.parquet';rows=pq.read_table(path).to_pylist()
    rows[4]['action'][6]=7.
    pq.write_table(pa.Table.from_pylist(rows),path)
    with pytest.raises(ValueError,match='terminal action'):Joint7LeRobotDataset(dataset_root)


@pytest.fixture
def long_dataset_root(dataset_root):
    """A true float32 timestamp column through 120 s, plus nine short episodes."""
    manifest_path=dataset_root/'meta/export_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['episodes'][0]['frame_count']=3601
    metadata,numeric=[],[]
    offset=0
    for episode,entry in enumerate(manifest['episodes']):
        length=entry['frame_count']
        state=np.asarray(entry['initial_state_measured_rad_m'])*np.array([180/np.pi]*6+[100.])
        meta=dict(episode_index=episode,length=length,tasks=[manifest['instruction']],
                  dataset_from_index=offset,dataset_to_index=offset+length)
        for key in CAMERA_KEYS:
            meta.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':0,
                         f'videos/{key}/from_timestamp':offset/30,f'videos/{key}/to_timestamp':(offset+length)/30})
        metadata.append(meta)
        for frame in range(length):
            numeric.append(dict(index=offset+frame,episode_index=episode,frame_index=frame,task_index=0,
                timestamp=frame/30,**{'observation.state':state.tolist(),'action':state.tolist()}))
        offset+=length
    info_path=dataset_root/'meta/info.json';info=json.loads(info_path.read_text())
    info['total_frames']=offset;info_path.write_text(json.dumps(info))
    manifest_path.write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist(metadata),dataset_root/'meta/episodes/chunk-000/file-000.parquet')
    table=pa.Table.from_pylist(numeric)
    index=table.column_names.index('timestamp')
    table=table.set_column(index,'timestamp',table['timestamp'].cast(pa.float32()))
    pq.write_table(table,dataset_root/'data/chunk-000/file-000.parquet')
    return dataset_root


def test_long_float32_timestamps_preserve_all_frames(long_dataset_root):
    path=long_dataset_root/'data/chunk-000/file-000.parquet'
    timestamps=np.asarray(pq.read_table(path,columns=['timestamp'])['timestamp'].to_pylist())
    assert abs(timestamps[1924]-1924/30)>2e-6
    assert timestamps[3600]==120.
    dataset=Joint7LeRobotDataset(long_dataset_root)
    assert dataset._states.shape==(3646,7)
    assert len(dataset)==sum(dataset.episode_lengths.values())


@pytest.mark.parametrize('corruption',['one_frame','one_millisecond','one_ulp','nonfinite'])
def test_long_episode_timestamp_misalignment_rejected(long_dataset_root,corruption):
    path=long_dataset_root/'data/chunk-000/file-000.parquet'
    table=pq.read_table(path);index=table.column_names.index('timestamp')
    times=np.asarray(table['timestamp'].to_pylist(),dtype=np.float32)
    if corruption=='one_frame':times[3004]+=np.float32(1/30)
    elif corruption=='one_millisecond':times[3004]+=np.float32(.001)
    elif corruption=='one_ulp':times[3004]=np.nextafter(times[3004],np.float32(np.inf))
    else:times[3004]=np.nan
    table=table.set_column(index,'timestamp',pa.array(times,pa.float32()))
    pq.write_table(table,path)
    with pytest.raises(ValueError,match='timestamps'):Joint7LeRobotDataset(long_dataset_root)


def test_generic_schema_keeps_same_normalization_and_rejects_unit_changes():
    from pressb.models.vlajepa_joint import validate_joint_schema
    from starVLA.training.train_table_buttons import action_schema
    from pressb.training.vlajepa_joint7_data import NORMALIZATION
    from types import SimpleNamespace
    schema=action_schema(SimpleNamespace(action_dim=7,normalization=NORMALIZATION),{},schema_name='vlajepa.joint7.v1')
    assert validate_joint_schema(schema,NORMALIZATION)==7
    schema['dataset_units']=['rad']*6+['m']
    with pytest.raises(ValueError):validate_joint_schema(schema,NORMALIZATION)


def test_generic_checkpoint_preserves_task_instruction_for_inference(tmp_path):
    import yaml
    from types import SimpleNamespace
    from pressb.models.vlajepa_joint import read_checkpoint_metadata
    from pressb.runtime.contract import fingerprint
    from starVLA.training.train_table_buttons import action_schema
    from pressb.training.vlajepa_joint7_data import NORMALIZATION
    run=tmp_path/'run';checkpoint=run/'checkpoints/best';checkpoint.mkdir(parents=True)
    (run/'processor').mkdir();(checkpoint/'model.pt').write_bytes(b'test-only')
    instruction='Put every object on the table into the box.'
    split=dict(source_export_manifest_sha256='a'*64,camera_keys=list(CAMERA_KEYS),task_id='cleanup',instruction=instruction)
    framework=dict(qwenvl=dict(init_from_config=True),vj2_model=dict(init_from_config=True),
                   action_model=dict(state_dim=7,action_dim=7,action_horizon=7))
    contract=dict(seed=42,framework=framework,data=dict(resolution_size=224),trainer={},
                  normalization=NORMALIZATION,task_id='cleanup',instruction=instruction)
    provenance=dict(dataset_sha256='a'*64,split_sha256=fingerprint(split),training_fingerprint=fingerprint(contract),
                    task_id='cleanup',instructions=[instruction],dataset_group='cleanup')
    schema=action_schema(SimpleNamespace(action_dim=7,normalization=NORMALIZATION),provenance,schema_name='vlajepa.joint7.v1')
    manifest=dict(complete=True,schema='vlajepa.joint7.v1',step=1,model_bytes=9,**provenance)
    for path,value in [('action_schema.json',schema),('normalization.json',NORMALIZATION),('split.json',split),
                       ('training_contract.json',dict(contract=contract,fingerprint=fingerprint(contract))),
                       ('checkpoints/best/manifest.json',manifest)]:
        (run/path).write_text(json.dumps(value))
    (run/'config.yaml').write_text(yaml.safe_dump(dict(run_id='cleanup',seed=42,framework=framework,
        datasets=dict(vla_data=contract['data']),trainer={})))
    assert read_checkpoint_metadata(run)['allowed_instructions']==[instruction]
    schema['instructions']=['Press the red button.'];(run/'action_schema.json').write_text(json.dumps(schema))
    with pytest.raises(ValueError,match='trained task'):read_checkpoint_metadata(run)
