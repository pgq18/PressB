#!/usr/bin/env python3
"""Compare paired collector attempts, including rejected attempts, without Isaac.

Physics JSONL is streamed and videos are hashed without decoding. Numeric
comparisons are by recorded index, never retimed. Array shape mismatches skip
array arithmetic; JSONL length mismatches explicitly report unmatched rows and
label any paired-prefix statistics as partial.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def raw_directory(path):
    path = path.resolve()
    return path/'raw' if (path/'raw').is_dir() else path


def index_attempts(raw):
    result = {}
    paths = list((raw/'episodes').glob('*/metadata.json'))
    paths += list((raw/'rejected').glob('*/metadata.json'))
    paths += list((raw/'rejected').glob('*.json'))
    for path in paths:
        metadata = read(path)
        if 'attempt_index' not in metadata:
            continue
        attempt = int(metadata['attempt_index'])
        if attempt in result:
            raise ValueError(f'Duplicate completed attempt {attempt}: {path}, {result[attempt]["metadata_path"]}')
        planning_rejection = path.parent == raw/'rejected'
        kind = ('planning_rejected' if planning_rejection else
                'accepted' if metadata.get('success') is True else 'execution_rejected')
        result[attempt] = dict(kind=kind, metadata=metadata, metadata_path=path,
            directory=None if planning_rejection else path.parent)
    return result


def numeric_array(left, right):
    left, right = np.asarray(left), np.asarray(right)
    result = dict(shape_left=list(left.shape), shape_right=list(right.shape),
        dtype_left=str(left.dtype), dtype_right=str(right.dtype),
        shape_equal=left.shape == right.shape, all_exact=False)
    if left.shape != right.shape:
        result['comparison_skipped_reason'] = 'Shape/length mismatch; no truncation or prefix array comparison performed.'
        return result
    if left.dtype.kind not in 'biuf' or right.dtype.kind not in 'biuf':
        raise ValueError('Expected real numeric array')
    a, b = left.astype(np.float64), right.astype(np.float64)
    equal = np.equal(a, b)
    finite = np.isfinite(a) & np.isfinite(b)
    errors = np.abs(a[finite]-b[finite])
    result.update(elements=int(a.size), exact_elements=int(equal.sum()),
        exact_fraction=float(equal.mean()) if a.size else 1.,
        all_exact=bool(equal.all()), nonfinite_left=int((~np.isfinite(a)).sum()),
        nonfinite_right=int((~np.isfinite(b)).sum()),
        max_abs_error=float(errors.max()) if errors.size else None,
        mean_abs_error=float(errors.mean()) if errors.size else None,
        rmse=float(np.sqrt(np.mean(errors**2))) if errors.size else None)
    if a.ndim >= 1:
        row_equal=equal.reshape((a.shape[0],-1)).all(axis=1) if a.shape[0] else np.zeros(0,bool)
        result.update(rows=int(a.shape[0]), exact_rows=int(row_equal.sum()),
                      exact_row_fraction=float(row_equal.mean()) if a.shape[0] else 1.)
    if a.ndim == 2 and a.shape[1] == 7:
        result['per_coordinate_max_abs_error'] = [float(np.abs(a[:,i]-b[:,i]).max()) if len(a) and np.all(finite[:,i]) else None for i in range(7)]
        result['coordinate_units'] = ['rad']*6+['m_full_gripper_width']
    return result


class TreeComparison:
    """Accumulate strict leaf equality and numeric errors by stable field path."""
    def __init__(self):
        self.rows = self.exact_rows = self.leaves = self.exact_leaves = 0
        self.structure_mismatches = 0
        self.fields = {}
        self.examples = []

    def mismatch(self, path, left, right, row):
        if len(self.examples) < 5:
            self.examples.append(dict(row_index=row, path=path, left=left, right=right))

    def add(self, left, right, row=0):
        self.rows += 1
        equal = self._visit(left,right,'',row)
        self.exact_rows += equal
        return equal

    def _visit(self, left, right, path, row):
        if isinstance(left,dict) and isinstance(right,dict):
            if left.keys()!=right.keys():
                self.structure_mismatches += 1
                self.mismatch(path,sorted(left),sorted(right),row)
            same = left.keys()==right.keys()
            for key in sorted(left.keys() & right.keys()):
                same = self._visit(left[key],right[key],f'{path}.{key}' if path else key,row) and same
            return same
        if isinstance(left,list) and isinstance(right,list):
            if len(left)!=len(right):
                self.structure_mismatches += 1
                self.mismatch(path,{'length':len(left)},{'length':len(right)},row)
                return False
            return all([self._visit(a,b,f'{path}[{i}]',row) for i,(a,b) in enumerate(zip(left,right))])
        if isinstance(left,(dict,list)) or isinstance(right,(dict,list)):
            self.structure_mismatches += 1
            self.mismatch(path,str(type(left)),str(type(right)),row)
            return False
        numeric = (isinstance(left,Real) and not isinstance(left,bool)
                   and isinstance(right,Real) and not isinstance(right,bool))
        equal = left==right and (numeric or type(left)==type(right))
        self.leaves += 1;self.exact_leaves += equal
        field=self.fields.setdefault(path,dict(values=0,exact_values=0,numeric_values=0,
            numeric_exact_values=0,max_abs_error=None,sum_abs_error=0.,sum_squared_error=0.,nonfinite_pairs=0))
        field['values']+=1;field['exact_values']+=equal
        if numeric:
            if math.isfinite(left) and math.isfinite(right):
                error=abs(float(left)-float(right))
                field['numeric_values']+=1;field['numeric_exact_values']+=equal
                field['max_abs_error']=max(field['max_abs_error'] or 0.,error)
                field['sum_abs_error']+=error;field['sum_squared_error']+=error*error
            else:
                field['nonfinite_pairs']+=1
        if not equal:self.mismatch(path,left,right,row)
        return bool(equal)

    def report(self):
        fields={}
        for path,value in sorted(self.fields.items()):
            n=value['numeric_values']
            fields[path]={k:v for k,v in value.items() if not k.startswith('sum_')}
            fields[path].update(exact_fraction=value['exact_values']/value['values'],
                mean_abs_error=value['sum_abs_error']/n if n else None,
                rmse=math.sqrt(value['sum_squared_error']/n) if n else None)
        numeric_count=sum(f['numeric_values'] for f in fields.values())
        numeric_exact=sum(f['numeric_exact_values'] for f in fields.values())
        errors=[f['max_abs_error'] for f in fields.values() if f['max_abs_error'] is not None]
        return dict(paired_rows=self.rows,exact_rows=self.exact_rows,
            exact_row_fraction=self.exact_rows/self.rows if self.rows else None,
            compared_leaves=self.leaves,exact_leaves=self.exact_leaves,
            exact_leaf_fraction=self.exact_leaves/self.leaves if self.leaves else None,
            numeric_values=numeric_count,numeric_exact_values=numeric_exact,
            numeric_exact_fraction=numeric_exact/numeric_count if numeric_count else None,
            max_abs_error=max(errors) if errors else None,
            structure_mismatches=self.structure_mismatches,
            all_exact=self.rows==self.exact_rows and self.structure_mismatches==0,
            fields=fields,first_mismatches=self.examples)


def tree(left,right):
    comparison=TreeComparison();comparison.add(left,right)
    return comparison.report()


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(),bytes=path.stat().st_size)


def pose_errors(left,right,stats):
    for name in left.keys() & right.keys():
        a,b=left[name],right[name]
        if not all(k in o for o in (a,b) for k in ('position_world','orientation_wxyz')):
            continue
        p,q=a['position_world'],b['position_world']
        qa,qb=a['orientation_wxyz'],b['orientation_wxyz']
        if len(p)!=3 or len(q)!=3 or len(qa)!=4 or len(qb)!=4:
            continue
        norms=math.sqrt(sum(x*x for x in qa)*sum(x*x for x in qb))
        same_rotation=qa==qb or all(x==-y for x,y in zip(qa,qb))
        angle=0. if same_rotation else 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(qa,qb)))/norms)) if norms else None
        distance=math.sqrt(sum((x-y)**2 for x,y in zip(p,q)))
        s=stats.setdefault(name,dict(paired_rows=0,max_position_distance_m=0.,max_orientation_angle_rad=0.))
        s['paired_rows']+=1;s['max_position_distance_m']=max(s['max_position_distance_m'],distance)
        if angle is not None:s['max_orientation_angle_rad']=max(s['max_orientation_angle_rad'],angle)


def physics_comparison(left,right):
    fields=('state','objects','contacts','physical_outcome','applied_actions','termination','index_phase')
    groups={key:TreeComparison() for key in fields}
    left_rows=right_rows=paired=exact_rows=0
    poses={}
    missing=object()
    def extract(row):
        info=row['info']
        return dict(state=row['state'],objects=row['objects'],
            contacts=info.get('contacts'),physical_outcome=info.get('physical_outcome'),
            applied_actions=info.get('applied_actions'),
            termination=dict(terminated=row['terminated'],truncated=row['truncated'],
                             reason=info.get('termination_reason')),
            index_phase={k:row.get(k) for k in ('action_index','phase','recorded','validation_index')})
    with left.open() as a,right.open() as b:
        for index,(aline,bline) in enumerate(zip_longest(a,b,fillvalue=missing)):
            left_rows += aline is not missing;right_rows += bline is not missing
            if aline is missing or bline is missing:continue
            arow,brow=json.loads(aline),json.loads(bline)
            aa,bb=extract(arow),extract(brow)
            same=True
            for name in fields:
                same=groups[name].add(aa[name],bb[name],index) and same
            exact_rows+=same;paired+=1
            pose_errors(arow['objects'],brow['objects'],poses)
    same_length=left_rows==right_rows
    return dict(rows_left=left_rows,rows_right=right_rows,length_equal=same_length,
        paired_rows=paired,unmatched_rows_left=left_rows-paired,unmatched_rows_right=right_rows-paired,
        exact_compared_rows=exact_rows,exact_compared_row_fraction=exact_rows/paired if paired else None,
        all_compared_fields_exact=same_length and paired>0 and paired==exact_rows,
        numeric_metrics_scope='all rows' if same_length else 'PAIRED PREFIX ONLY; lengths differ and unmatched rows are explicitly excluded from numeric errors',
        alignment='Recorded index / 30 Hz time; no interpolation, DTW, resampling, or phase realignment.',
        groups={name:stats.report() for name,stats in groups.items()},object_pose_errors=poses)


def compare_attempt(index,left,right,hash_videos=True):
    a,b=left['metadata'],right['metadata']
    row=dict(attempt_index=index,kind_left=left['kind'],kind_right=right['kind'],
        metadata_left=str(left['metadata_path']),metadata_right=str(right['metadata_path']),
        kinds_equal=left['kind']==right['kind'],seed_equal=a.get('seed')==b.get('seed'),
        success_left=a.get('success'),success_right=b.get('success'),
        success_equal=a.get('success')==b.get('success'),
        env_id_left=a.get('env_id'),env_id_right=b.get('env_id'),
        numeric_comparison_notes=[])
    if left['kind']=='planning_rejected' or right['kind']=='planning_rejected':
        row['planning_rejection_comparison']=dict(stage_equal=a.get('stage')==b.get('stage'),
            error_equal=a.get('error')==b.get('error'),error_left=a.get('error'),error_right=b.get('error'))
        row['executed_trace_comparable']=False
        return row
    row['executed_trace_comparable']=True
    if a.get('env_id')!=b.get('env_id'):
        row['numeric_comparison_notes'].append('Physical slot differs: world-position errors may include clone translation. No silent coordinate correction is applied.')
    row['audit_passed_equal']=(a.get('audit') or {}).get('passed')==(b.get('audit') or {}).get('passed')
    row['audit']=tree(a.get('audit'),b.get('audit'))
    row['reset_objects']=tree((a.get('reset_spec') or {}).get('objects'),(b.get('reset_spec') or {}).get('objects'))
    ad,bd=left['directory'],right['directory']
    ap,bp=read(ad/'plan.json'),read(bd/'plan.json')
    row['plan']=dict(fps_left=ap.get('fps'),fps_right=bp.get('fps'),fps_equal=ap.get('fps')==bp.get('fps'),
        actions=numeric_array(ap['actions'],bp['actions']),phases=tree(ap.get('phases'),bp.get('phases')),
        noise=tree(ap.get('noise'),bp.get('noise')))
    with np.load(ad/'data.npz',allow_pickle=False) as an,np.load(bd/'data.npz',allow_pickle=False) as bn:
        row['data_npz']=dict(keys_left=sorted(an.files),keys_right=sorted(bn.files),
            keys_equal=set(an.files)==set(bn.files),missing_keys_left=sorted(set(bn.files)-set(an.files)),
            missing_keys_right=sorted(set(an.files)-set(bn.files)),
            arrays={key:numeric_array(an[key],bn[key]) for key in sorted(set(an.files)&set(bn.files))})
    row['physics']=physics_comparison(ad/'physics.jsonl',bd/'physics.jsonl')
    if hash_videos:
        row['videos']={}
        for camera in ('up','wrist'):
            pa,pb=ad/f'{camera}.mp4',bd/f'{camera}.mp4'
            ha,hb=sha256(pa) if pa.exists() else None,sha256(pb) if pb.exists() else None
            row['videos'][camera]=dict(left=ha,right=hb,sha256_equal=ha is not None and hb is not None and ha['sha256']==hb['sha256'])
    return row


def aggregate(rows,missing_left,missing_right):
    executed=[r for r in rows if r['executed_trace_comparable']]
    result=dict(paired_attempts=len(rows),paired_executed_attempts=len(executed),
        missing_left=missing_left,missing_right=missing_right,
        kind_mismatches=[r['attempt_index'] for r in rows if not r['kinds_equal']],
        success_mismatches=[r['attempt_index'] for r in rows if not r['success_equal']],
        audit_passed_mismatches=[r['attempt_index'] for r in executed if not r['audit_passed_equal']],
        plan_shape_mismatches=[r['attempt_index'] for r in executed if not r['plan']['actions']['shape_equal']],
        plan_fps_mismatches=[r['attempt_index'] for r in executed if not r['plan']['fps_equal']],
        data_npz_key_mismatches=[r['attempt_index'] for r in executed if not r['data_npz']['keys_equal']],
        physics_length_mismatches=[r['attempt_index'] for r in executed if not r['physics']['length_equal']],
        kinds_left=dict(Counter(r['kind_left'] for r in rows)),kinds_right=dict(Counter(r['kind_right'] for r in rows)))
    result['arrays']={}
    for label in ('plan_actions','data_state','data_action','data_sim_time'):
        values=[]
        for r in executed:
            a=r['plan']['actions'] if label=='plan_actions' else r['data_npz']['arrays'].get(label.removeprefix('data_'))
            if a is not None:values.append((r['attempt_index'],a))
        total=sum(a.get('elements',0) for _,a in values)
        exact=sum(a.get('exact_elements',0) for _,a in values)
        errors=[a['max_abs_error'] for _,a in values if a.get('max_abs_error') is not None]
        result['arrays'][label]=dict(compared_elements=total,exact_elements=exact,
            exact_fraction=exact/total if total else None,max_abs_error=max(errors) if errors else None,
            shape_mismatch_attempts=[i for i,a in values if not a['shape_equal']],
            nonexact_attempts=[i for i,a in values if not a['all_exact']])
    result['physics_groups']={}
    for name in ('state','objects','contacts','physical_outcome','applied_actions','termination','index_phase'):
        stats=[r['physics']['groups'][name] for r in executed]
        leaves=sum(s['compared_leaves'] for s in stats);exact=sum(s['exact_leaves'] for s in stats)
        errors=[s['max_abs_error'] for s in stats if s['max_abs_error'] is not None]
        result['physics_groups'][name]=dict(compared_leaves=leaves,exact_leaves=exact,
            exact_leaf_fraction=exact/leaves if leaves else None,max_abs_error=max(errors) if errors else None,
            nonexact_attempts=[r['attempt_index'] for r in executed if not r['physics']['groups'][name]['all_exact']],
            numeric_errors_use_paired_prefix_when_lengths_differ=True)
    result['video_hashes']={camera:dict(equal=sum(r.get('videos',{}).get(camera,{}).get('sha256_equal',False) for r in executed),
        compared=sum(camera in r.get('videos',{}) for r in executed)) for camera in ('up','wrist')}
    return result


def markdown(report):
    s=report['summary']
    out=['# Paired manipulation trace comparison','',
        f"Baseline: `{report['baseline']}`  \nCandidate: `{report['candidate']}`",'',
        f"Compared attempts 1–{report['attempt_prefix']}: {s['paired_attempts']} paired, {s['paired_executed_attempts']} with execution traces.",
        f"Missing baseline/candidate: {s['missing_left']} / {s['missing_right']}.",
        f"Outcome-kind differences: {s['kind_mismatches']}; success differences: {s['success_mismatches']}; physical-audit pass differences: {s['audit_passed_mismatches']}.",
        f"Plan shape differences: {s['plan_shape_mismatches']}; physics length differences: {s['physics_length_mismatches']}.",'',
        '| Array | Strict equal elements | Maximum absolute error | Shape mismatches |',
        '|---|---:|---:|---|']
    for name,v in s['arrays'].items():
        fraction='—' if v['exact_fraction'] is None else f"{100*v['exact_fraction']:.6f}%"
        error='—' if v['max_abs_error'] is None else f"{v['max_abs_error']:.9g}"
        out.append(f"| {name} | {fraction} | {error} | {v['shape_mismatch_attempts']} |")
    out+=['','| Physics field | Strict equal leaves | Maximum absolute numeric error | Nonexact attempts |',
          '|---|---:|---:|---|']
    for name,v in s['physics_groups'].items():
        fraction='—' if v['exact_leaf_fraction'] is None else f"{100*v['exact_leaf_fraction']:.6f}%"
        error='—' if v['max_abs_error'] is None else f"{v['max_abs_error']:.9g}"
        out.append(f"| {name} | {fraction} | {error} | {v['nonexact_attempts']} |")
    out+=['','Detailed JSON separates numeric fields and preserves units in field names; the aggregate maximum can mix units. '
          'Joint-state coordinates are six radians plus gripper width in metres; contacts are newtons. '
          'Quaternion components are compared strictly, with additional sign-invariant angular errors per object.',
          '','Array shape mismatches skip array arithmetic. Physics length mismatches are explicit: numeric errors then cover only paired rows; unmatched rows are never treated as equal. '
          'Rows are compared by control-time index without alignment or interpolation.',
          '','MP4 checks use file SHA-256 only, without decoding. Unequal hashes do not establish a pixel-level difference because encoding/container metadata can differ.', '']
    for camera,value in s['video_hashes'].items():out.append(f"- {camera}: {value['equal']}/{value['compared']} identical file hashes.")
    return '\n'.join(out)+'\n'


def compare(baseline,candidate,prefix,hash_videos=True):
    baseline,candidate=raw_directory(baseline),raw_directory(candidate)
    left,right=index_attempts(baseline),index_attempts(candidate)
    wanted=set(range(1,prefix+1))
    missing_left=sorted(wanted-set(left));missing_right=sorted(wanted-set(right))
    rows=[compare_attempt(i,left[i],right[i],hash_videos) for i in sorted(wanted & set(left) & set(right))]
    return dict(schema_version=1,baseline=str(baseline),candidate=str(candidate),attempt_prefix=prefix,
        attempts=rows,summary=aggregate(rows,missing_left,missing_right),
        semantics=dict(match='Global attempt_index, including planning and physical-audit rejections.',
            numeric='Strict equality plus absolute error; no tolerance changes or silent truncation.',
            physics='Same control-time index. JSONL length mismatch reports unmatched rows and labels paired-prefix numeric statistics.',
            videos='File SHA-256, no video decoding. Unequal hash does not imply changed decoded pixels.',
            determinism='PhysX and renderer need not be bitwise reproducible across runs; differences are measured, not hidden.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path,help='Baseline case or raw directory')
    parser.add_argument('--candidate',required=True,type=Path,help='Candidate case or raw directory')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--attempt-prefix',type=int,default=16)
    parser.add_argument('--skip-video-hashes',action='store_true')
    args=parser.parse_args()
    if args.attempt_prefix<1:parser.error('Attempt prefix must be positive')
    report=compare(args.baseline,args.candidate,args.attempt_prefix,not args.skip_video_hashes)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(markdown(report))
    print(json.dumps(report['summary'],allow_nan=False))


if __name__=='__main__':
    main()
