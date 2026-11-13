"""Task-independent joint7 LeRobot v3 reader for bundled VLA-JEPA SFT.

Consumes PressB's explicitly versioned export contract, never guesses units or
relabels a task as table-buttons. Decoding/caching and fixed joint normalization
reuse the native model reader; episode partitioning is independent of frames.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from starVLA.dataloader.piper_lerobot import (
    PiperLeRobotDataset, _require, _sha256, _canonical_sha, _split_manifest,
)
from starVLA.dataloader.table_buttons_lerobot import (
    TableButtonsLeRobotDataset, CAMERA_KEYS, NORMALIZATION, normalize_joint7, denormalize_joint7,
)

SEMANTICS = dict(state_units=['deg']*6+['cm'], action_units=['deg']*6+['cm'],
                 gripper_representation='full_width', action='next_interval_absolute_joint_target',
                 terminal_action='repeat_last_applied')


class Joint7LeRobotDataset(PiperLeRobotDataset):
    """Arbitrary single-task episode count, with an immutable 90/10 split."""
    camera_keys = CAMERA_KEYS
    action_dim = state_dim = 7
    normalize = staticmethod(normalize_joint7)
    denormalize = staticmethod(denormalize_joint7)
    _load_numeric = TableButtonsLeRobotDataset._load_numeric

    def _timestamps_match(self, values, frames):
        """Allow float32 storage rounding, never a misplaced 30 Hz sample.

        Beyond 64 s the float32 half-ULP exceeds the historical fixed 2 us
        tolerance. Scale only with representational precision, not duration or
        frame period. The tiny floor handles timestamps very near zero.
        """
        expected=np.asarray(frames,dtype=np.float64)/self.fps
        actual=np.asarray(values,dtype=np.float64)
        half_ulp=np.spacing(expected.astype(np.float32)).astype(np.float64)*.500001
        return (actual.shape==expected.shape and np.isfinite(actual).all()
                and bool(np.all(np.abs(actual-expected)<=np.maximum(half_ulp,1e-9))))

    def __init__(self, root, split='train', split_manifest=None, action_horizon=7,
                 video_horizon=8, resolution=224, seed=42, sample_stride=1,
                 frame_cache_size=256, decoder_cache_size=4, image_transforms=None):
        self.root = Path(root).resolve()
        _require(split in ('train', 'val'), 'split must be train or val')
        _require(type(seed) is int and seed >= 0, 'seed must be a nonnegative integer')
        for key, value in [('action_horizon',action_horizon),('video_horizon',video_horizon),
                           ('sample_stride',sample_stride),('decoder_cache_size',decoder_cache_size)]:
            _require(type(value) is int and value > 0, f'{key} must be a positive integer')
        _require(type(frame_cache_size) is int and frame_cache_size >= 0, 'Invalid frame cache size')
        self.split, self.sample_stride = split, sample_stride
        self.action_horizon, self.video_horizon = action_horizon, video_horizon
        self.frame_cache_size, self.decoder_cache_size = frame_cache_size, decoder_cache_size
        self.image_transforms = image_transforms if split == 'train' else None
        if isinstance(resolution, int): resolution = (resolution, resolution)
        _require(len(resolution)==2 and all(type(v) is int and v > 0 for v in resolution), 'Invalid resolution')
        self.resolution = tuple(resolution)
        info_path, manifest_path = self.root/'meta/info.json', self.root/'meta/export_manifest.json'
        self.info, self.manifest = json.loads(info_path.read_text()), json.loads(manifest_path.read_text())
        self.export_manifest_sha256 = _sha256(manifest_path)
        self.fps = 30
        _require(self.info.get('codebase_version')=='v3.0' and self.info.get('fps')==30
                 and self.info.get('total_tasks')==1, 'Expected single-task LeRobot v3 at 30 Hz')
        for key in ('observation.state', 'action'):
            _require(self.info['features'][key]['shape']==[7], f'{key} must contain seven coordinates')
        _require(self.manifest.get('schema_version')==1, 'Unsupported export manifest version')
        semantic = self.manifest.get('state_action_semantics', {})
        _require(all(semantic.get(k)==v for k,v in SEMANTICS.items()), 'Unknown joint7 units or action timing semantics')
        self.task_id = self.manifest.get('task_id')
        self.instruction = self.manifest.get('instruction')
        _require(isinstance(self.task_id,str) and self.task_id.strip()
                 and isinstance(self.instruction,str) and self.instruction.strip(), 'Task identity and instruction required')
        self.group = self.task_id
        self.normalization = deepcopy(NORMALIZATION)
        entries = self.manifest.get('episodes', [])
        count = self.info['total_episodes']
        _require(type(count) is int and count >= 2 and len(entries)==count, 'At least two complete episodes required')
        for index, entry in enumerate(entries):
            _require(entry.get('episode_index')==index and entry.get('success') is True
                     and entry.get('instruction')==self.instruction, 'Episode identity, success or instruction mismatch')
            _require(type(entry.get('frame_count')) is int and entry['frame_count']>=2, 'Invalid episode frame count')
            _require(type(entry.get('seed')) is int and entry['seed']>=0, 'Invalid collection seed')
        _require(len({e['seed'] for e in entries})==count, 'Duplicate collection seeds')
        val_count = max(1, min(count-1, round(count*.1)))
        permutation = np.random.default_rng(seed).permutation(count).tolist()
        expected = dict(schema_version=1, kind='pressb_joint7_episode_split',seed=seed,
            source_export_manifest_sha256=self.export_manifest_sha256,task_id=self.task_id,
            instruction=self.instruction,camera_keys=list(CAMERA_KEYS),
            train_source_episode_ids=sorted(permutation[val_count:]),
            val_source_episode_ids=sorted(permutation[:val_count]), counts=dict(train=count-val_count,val=val_count))
        self.split_manifest = _split_manifest(expected, split_manifest, self.root)
        self.split_manifest_sha256 = _canonical_sha(self.split_manifest)
        self.source_episode_count = count
        self.source_episode_ids = self.split_manifest[f'{split}_source_episode_ids']
        selected = set(self.source_episode_ids)
        episodes = [row for p in sorted((self.root/'meta/episodes').rglob('*.parquet')) for row in pq.read_table(p).to_pylist()]
        episodes.sort(key=lambda row:row['episode_index'])
        _require(len(episodes)==count, 'Episode metadata count mismatch')
        self._episodes, self._all_episodes, self._anchors, self._ends = {}, [], [], []
        end = 0
        for index,(meta,entry) in enumerate(zip(episodes,entries)):
            length = entry['frame_count']
            _require(meta['episode_index']==index and meta['length']==length and meta['dataset_from_index']==end
                     and meta['dataset_to_index']==end+length and meta['tasks']==[self.instruction], 'Episode boundaries or instruction mismatch')
            ep = dict(meta, source_episode_id=index, task=self.instruction, entry=entry, videos={})
            for key in CAMERA_KEYS:
                prefix=f'videos/{key}/'
                first,last=[float(meta[prefix+k])*self.fps for k in ('from_timestamp','to_timestamp')]
                _require(np.isfinite([first,last]).all() and first>=0 and abs(first-round(first))<1e-3
                         and abs(last-first-length)<1e-3, 'Video interval differs from numeric episode')
                path=self.root/self.info['video_path'].format(video_key=key,chunk_index=meta[prefix+'chunk_index'],file_index=meta[prefix+'file_index'])
                _require(path.is_file(), f'Missing video {path}')
                ep['videos'][key]=(str(path),round(first))
            self._all_episodes.append(ep)
            if index in selected:
                self._episodes[index]=ep
                anchors=list(range(0,length,sample_stride))
                if anchors[-1]!=length-1: anchors.append(length-1)
                self._anchors.append((index,np.asarray(anchors,dtype=np.int64)))
                self._ends.append((self._ends[-1] if self._ends else 0)+len(anchors))
            end+=length
        _require(end==self.info['total_frames'], 'Episode metadata does not span all frames')
        self.episode_lengths={i:e['length'] for i,e in self._episodes.items()}
        self._load_numeric(end)
        self.fingerprint = dict(adapter='PressBJoint7LeRobotDataset',version=1,task_id=self.task_id,
            source_export_manifest_sha256=self.export_manifest_sha256,source_info_sha256=_sha256(info_path),
            source_numeric_sha256=self._numeric_sha256,normalization=self.normalization,
            split_manifest_sha256=self.split_manifest_sha256,camera_keys=list(CAMERA_KEYS),fps=self.fps,
            action_horizon=action_horizon,video_horizon=video_horizon,resolution=list(self.resolution),sample_stride=sample_stride)
        self.fingerprint_sha256 = _canonical_sha(self.fingerprint)
        self._pid, self._frames, self._decoders = os.getpid(), OrderedDict(), OrderedDict()

    def sample_metadata(self,index):
        if not isinstance(index,(int,np.integer)) or not 0<=index<len(self): raise IndexError(index)
        part=bisect_right(self._ends,int(index))
        offset=int(index)-(self._ends[part-1] if part else 0)
        episode,anchors=self._anchors[part]; frame=int(anchors[offset]); meta=self._episodes[episode]
        return dict(source_episode_id=episode,episode_index=episode,frame_index=frame,
                    dataset_index=meta['dataset_from_index']+frame,task_id=self.task_id,timestamp=frame/self.fps)

    def get_frame(self,source_episode_id,frame_index):
        if source_episode_id not in self._episodes: raise KeyError('Episode outside selected split')
        ep=self._episodes[source_episode_id]
        if not isinstance(frame_index,(int,np.integer)) or not 0<=frame_index<ep['length']: raise IndexError(frame_index)
        i,final=int(frame_index),ep['length']-1
        a_req,v_req=i+np.arange(self.action_horizon),i+np.arange(self.video_horizon)
        ai=ep['dataset_from_index']+np.minimum(a_req,final); vi=np.minimum(v_req,final)
        views=[]
        for key in CAMERA_KEYS:
            path,start=ep['videos'][key]
            views.append(self._augment_video(self._video_frames(path,start+vi),sample_key=(int(source_episode_id),i,key)))
        return dict(image=[Image.fromarray(v[0].copy()) for v in views],video=np.stack(views),
            state=self._states[ep['dataset_from_index']+i:ep['dataset_from_index']+i+1].copy(),
            action=self._actions[ai].copy(),lang=self.instruction,action_is_pad=a_req>final,video_is_pad=v_req>final,
            source_episode_id=source_episode_id,episode_index=source_episode_id,frame_index=i,
            dataset_index=ep['dataset_from_index']+i,task_id=self.task_id,group=self.group,timestamp=i/self.fps,
            video_timestamps=(vi/self.fps).astype(np.float64))
