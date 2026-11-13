"""Single-task, arbitrary-size joint7 SFT using the common native trainer."""
from __future__ import annotations
import hashlib
import json

SCHEMA = 'vlajepa.joint7.v1'


def build_contract(cfg, normalization, split_manifest):
    from starVLA.training import train_piper as shared
    from starVLA.training import train_table_buttons as native
    if cfg['framework']['action_model']['action_dim']!=7 or cfg['framework']['action_model']['state_dim']!=7:
        raise ValueError('Generic joint7 SFT requires seven measured state and action coordinates')
    counts=split_manifest['counts']
    if min(counts.values())<1:
        raise ValueError('Both episode partitions must be nonempty')
    contract,_=shared.training_contract(cfg)
    contract.update(schema=SCHEMA,normalization=normalization,
        task_id=split_manifest['task_id'],instruction=split_manifest['instruction'],
        source_episode_count=sum(counts.values()),split_counts=counts,
        split_rule='episode permutation by configured seed; round(10%) validation, remaining train',
        initialization='official pretrained; four joint7 state/action interface tensors reinitialized',
        checkpoint_policy=native.configured_checkpoint_policy(cfg),
        validation_selection_weight=native.GRIPPER_SCORE_WEIGHT,
        validation='offline held-out joint prediction; not simulated task success')
    warm = native.warm_start_configuration(cfg)
    if warm is not None:
        contract['trainer'].update(warm)
        contract.update(initialization='model-only warm start; all joint7 tensors retained; fresh optimizer and schedule',
                        warm_start_optimizer_state='not loaded', warm_start_step_semantics='new run counts additional optimizer updates from zero')
    return contract,hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_anchors(dataset,anchors):
    if not anchors or len(anchors)!=len(set(anchors)):
        raise ValueError('Validation requires unique episode/frame anchors')
    if any(e not in dataset.episode_lengths or not 0<=f<dataset.episode_lengths[e] for e,f in anchors):
        raise ValueError('Validation anchor lies outside held-out episodes')
    return len(anchors)


def main():
    from pressb.training.vlajepa_joint7_data import Joint7LeRobotDataset
    from starVLA.training import train_table_buttons as native
    native.main(dataset_class=Joint7LeRobotDataset,contract_builder=build_contract,
                anchor_validator=validate_anchors,schema_name=SCHEMA)


if __name__=='__main__':
    main()
