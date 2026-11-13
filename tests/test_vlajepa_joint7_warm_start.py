"""Strict model-only continuation keeps learned interfaces and starts new updates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pressb.model_sources import activate_source
activate_source('vlajepa')
from starVLA.training import train_table_buttons as trainer
from pressb.training.vlajepa_joint7 import build_contract


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class Tokenizer:
    pad_token_id=0;bos_token_id=1;eos_token_id=2
    def get_vocab(self):return {'pad':0,'bos':1,'eos':2,'action':3}


def tiny_model():
    model=torch.nn.Module()
    for name in ('qwen_vl_interface','vj_encoder','vj_predictor'):
        setattr(model,name,torch.nn.Linear(2,2))
    model.qwen_vl_interface.processor=SimpleNamespace(tokenizer=Tokenizer())
    model.action_model=torch.nn.Module()
    for branch,layer,input_size,output_size in (('action_encoder','layer1',7,4),('state_encoder','layer1',7,4),('action_decoder','layer2',4,7)):
        module=torch.nn.Module();setattr(module,layer,torch.nn.Linear(input_size,output_size));setattr(model.action_model,branch,module)
    return model


@pytest.fixture
def source(tmp_path):
    run=tmp_path/'source';checkpoint=run/'checkpoints/last_step_019167';checkpoint.mkdir(parents=True)
    model=tiny_model()
    with torch.no_grad():
        for index,p in enumerate(model.parameters()):p.fill_(.123456789+index*.0123456789)
    path=checkpoint/'model.pt';torch.save(model.state_dict(),path)
    (checkpoint/'training.pt').write_bytes(b'not a valid optimizer state; must never be loaded')
    normalization={'scale':[180]*6+[8],'kind':'fixed_joint7_scale'}
    split=dict(source_export_manifest_sha256='a'*64,task_id='cube_stack',camera_keys=['observation.images.up','observation.images.wrist'])
    contract=dict(seed=42,normalization=normalization)
    provenance=dict(dataset_sha256='a'*64,split_sha256=fingerprint(split),dataset_fingerprint_sha256='c'*64,
                    training_fingerprint=fingerprint(contract),task_id='cube_stack')
    manifest=dict(complete=True,schema='vlajepa.joint7.v1',step=19167,model_bytes=path.stat().st_size,**provenance)
    schema=dict(schema='vlajepa.joint7.v1',state_dim=7,action_dim=7,normalization=normalization,**provenance)
    for relative,value in [('checkpoints/last_step_019167/manifest.json',manifest),('normalization.json',normalization),
        ('action_schema.json',schema),('training_contract.json',dict(contract=contract,fingerprint=fingerprint(contract))),
        ('split.json',split),('processor/tokenizer_config.json',{})]:save(run/relative,value)
    settings=dict(warm_start_checkpoint=str(path),warm_start_sha256=trainer.shared.sha256(path),warm_start_step=19167)
    return dict(run=run,path=path,model=model,normalization=normalization,provenance=provenance,cfg={'trainer':settings})


def test_no_warm_start_preserves_default_and_partial_fields_rejected():
    assert trainer.warm_start_configuration({'trainer':{}}) is None
    with pytest.raises(ValueError,match='together'):
        trainer.warm_start_configuration({'trainer':{'warm_start_step':1}})


@pytest.mark.parametrize('key,value',[
    ('warm_start_checkpoint','relative/model.pt'),('warm_start_checkpoint','/absolute/checkpoint'),
    ('warm_start_sha256','incorrect'),('warm_start_sha256','A'*64),
    ('warm_start_step',0),('warm_start_step',True),('warm_start_step',1.5)])
def test_explicit_warm_start_identity_is_validated(source,key,value):
    cfg=deepcopy(source['cfg']);cfg['trainer'][key]=value
    with pytest.raises(ValueError):trainer.warm_start_configuration(cfg)


def test_model_only_continuation_cannot_be_combined_with_exact_resume(source):
    with pytest.raises(ValueError,match='mutually exclusive'):
        trainer.warm_start_configuration(source['cfg'],resume=source['path'].parent)


def test_verified_metadata_retains_source_identity_and_uses_parent_processor(source,tmp_path):
    report=trainer.inspect_warm_start(source['cfg'],provenance=source['provenance'],normalization=source['normalization'],output=tmp_path/'new')
    assert report['source_step']==19167 and report['processor_path']==str(source['run']/'processor')
    assert report['checkpoint_sha256']==source['cfg']['trainer']['warm_start_sha256']
    assert not report['optimizer_state_loaded'] and not report['scheduler_state_loaded'] and not report['interfaces_reinitialized']


def test_source_run_cannot_be_overwritten(source):
    with pytest.raises(ValueError,match='new run'):
        trainer.inspect_warm_start(source['cfg'],output=source['run'])


@pytest.mark.parametrize('field,value',[('complete',False),('schema','vlajepa.table_buttons.joint6.v1'),('step',19000),('model_bytes',1)])
def test_incomplete_wrong_schema_or_step_checkpoint_is_rejected(source,field,value):
    path=source['path'].parent/'manifest.json';manifest=json.loads(path.read_text());manifest[field]=value;save(path,manifest)
    with pytest.raises(ValueError):trainer.inspect_warm_start(source['cfg'])


def test_changed_model_and_changed_split_are_rejected(source):
    cfg=deepcopy(source['cfg']);cfg['trainer']['warm_start_sha256']='f'*64
    with pytest.raises(ValueError,match='SHA256'):trainer.inspect_warm_start(cfg)
    split=json.loads((source['run']/'split.json').read_text());split['task_id']='different';save(source['run']/'split.json',split)
    with pytest.raises(ValueError,match='source split'):trainer.inspect_warm_start(source['cfg'])


def test_different_dataset_split_and_normalization_are_rejected(source):
    for key in ('dataset_sha256','split_sha256','dataset_fingerprint_sha256','task_id'):
        provenance=dict(source['provenance'],**{key:'different'})
        with pytest.raises(ValueError,match='data contract'):
            trainer.inspect_warm_start(source['cfg'],provenance=provenance)
    with pytest.raises(ValueError,match='normalization'):
        trainer.inspect_warm_start(source['cfg'],normalization={'scale':[1]*7})


def test_missing_processor_is_rejected(source):
    (source['run']/'processor/tokenizer_config.json').unlink()
    with pytest.raises(ValueError,match='processor'):trainer.inspect_warm_start(source['cfg'])


def test_full_warm_load_keeps_every_interface_and_source_precision(source,monkeypatch):
    from transformers import AutoProcessor
    model=tiny_model().to(dtype=torch.bfloat16)
    report=trainer.inspect_warm_start(source['cfg'])
    monkeypatch.setattr(AutoProcessor,'from_pretrained',lambda *a,**kw:SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(trainer,'initialize_interfaces',lambda *a:(_ for _ in ()).throw(AssertionError('must never reset learned interfaces')))
    loading=trainer.load_model_only_warm_start(model,report)
    assert loading['initialization']=='model_only_warm_start'
    for name,value in model.state_dict().items():
        assert value.dtype==torch.float32 and torch.equal(value,source['model'].state_dict()[name])
    assert set(loading['step_zero_interfaces'])==trainer.INTERFACE_KEYS
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    assert len(optimizer.state)==0
    loss=sum(p.square().sum() for p in model.parameters());loss.backward();optimizer.step()
    assert all(int(state['step'])==1 for state in optimizer.state.values())


def contract_cfg():
    import yaml
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/model_training/vlajepa_joint7_generic.yaml').read_text())
    cfg['trainer'].update(max_train_steps=19167,num_warmup_steps=958)
    return cfg


def test_contract_includes_warm_source_only_when_enabled(source):
    cfg=contract_cfg();split=dict(counts={'train':450,'val':50},task_id='cube_stack',instruction='Stack the red cube.')
    original,original_fingerprint=build_contract(cfg,source['normalization'],split)
    assert 'warm_start_step' not in original['trainer'] and 'four joint7' in original['initialization']
    cfg['trainer'].update(source['cfg']['trainer'])
    continued,continued_fingerprint=build_contract(cfg,source['normalization'],split)
    assert continued_fingerprint!=original_fingerprint
    assert continued['trainer']['warm_start_step']==19167
    assert continued['trainer']['target_epochs']==3
    assert 'fresh optimizer' in continued['initialization']
    assert continued['warm_start_step_semantics']=='new run counts additional optimizer updates from zero'
