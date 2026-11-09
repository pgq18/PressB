"""Real small native action heads: gradients, frozen boundaries and transfer.

These checks use randomly initialized small architectures, not PAP checkpoint
rollouts, and therefore make no claim about task success or RL improvement.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_native(code, source, tmp_path, *args):
    dependencies = ("torch", "transformers", "omegaconf", "diffusers") if source.name == "vlajepa" else ("torch", "transformers", "torchdiffeq")
    for dependency in dependencies:
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Native action-expert check requires {dependency}")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(source)]),
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dimension,state_dimension", [(9, 9), (6, 7), (7, 7)])
def test_vlajepa_native_expert_updates_only_head_and_matches_euler(dimension, state_dimension, tmp_path):
    code = r'''
import copy, sys
from types import SimpleNamespace
import torch
from torch import nn
from omegaconf import OmegaConf
from starVLA.model.modules.action_model import GR00T_ActionHeader as native
from pressb.models.flow_experts import VLAJEPAFlowExpert, create_vlajepa_expert
from pressb.models.vlajepa_backend import flow_from_context
from pressb.runtime.denoising import GaussianChainPolicy
torch.set_num_threads(1)
torch.manual_seed(42)
dim, state_dim = map(int, sys.argv[1:3])
native.DiTConfig['test_cpu'] = dict(input_embedding_dim=16, attention_head_dim=8, num_attention_heads=2)
cfg = OmegaConf.create(dict(framework=dict(action_model=dict(
    action_model_type='test_cpu', hidden_size=16, action_dim=dim,
    state_dim=state_dim, action_horizon=7, future_action_window_size=6,
    num_inference_timesteps=3, num_target_vision_tokens=4, add_pos_embed=True,
    max_seq_len=32, noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999,
    num_timestep_buckets=1000, diffusion_model_cfg=dict(cross_attention_dim=12,
    num_layers=2, output_dim=16, dropout=.2, final_dropout=False,
    interleave_self_attention=True, norm_type='ada_norm', positional_embeddings=None)))))
model = nn.Module()
model.encoder = nn.Linear(12,12)
model.action_model = native.FlowmatchingActionHead(cfg)
model2 = copy.deepcopy(model)
backend = SimpleNamespace(frozen=SimpleNamespace(model=model)) if dim == 9 else SimpleNamespace(model=model,dimension=dim)
expert = create_vlajepa_expert(backend, {'transition_std': .04})
expert2 = VLAJEPAFlowExpert(model2)
assert all(name.startswith('head.') for name,_ in expert.named_parameters())
assert all(p.requires_grad for p in expert.parameters())
assert not any(p.requires_grad for p in model.encoder.parameters())
assert len(expert.state_dict()) == len(model.action_model.state_dict())
with torch.inference_mode():
    contexts = [SimpleNamespace(tokens=torch.randn(1,4,12),state=torch.randn(1,1,state_dim)) for _ in range(2)]
    packed = expert.pack_contexts(contexts)
assert all(not v.is_inference() and not v.requires_grad and v.device.type=='cpu' for v in packed.values())
assert all(torch.equal(v, expert.collate_conditions([{k:t[:1] for k,t in packed.items()},{k:t[1:] for k,t in packed.items()}])[key]) for key,v in packed.items())
x = torch.randn(2,7,dim)
expected = flow_from_context(model.action_model, packed['tokens'], packed['state'], x)
actual=x.clone()
for k in range(3):
    actual,std = expert.transition(actual,torch.full((2,),k),packed)
    assert torch.all(std == .04) and actual.dtype == torch.float32
torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
chain_policy=GaussianChainPolicy(expert)
sample=chain_policy.sample(packed,[11,19],initial_noise=x)
for k in range(3):
    score=chain_policy.log_prob(packed,sample['chain'][:,k],sample['chain'][:,k+1],k)
    torch.testing.assert_close(score,sample['logprobs'][:,k],atol=0,rtol=0)
assert chain_policy.sample(packed,[11,19],deterministic=True)['logprobs'] is None
if dim != 9:
    scale=torch.tensor([torch.pi]*6+([.08] if dim==7 else []))
    torch.testing.assert_close(expert.actions_from_latent(actual), actual*scale)
else:
    assert torch.equal(expert.actions_from_latent(actual),actual)
cached={k:v.clone() for k,v in packed.items()}
frozen=model.encoder.weight.clone()
before=model.action_model.action_decoder.layer2.weight.detach().clone()
expert.train(True)
assert not expert.head.training and not model.training
first,std=expert.transition(x,torch.tensor([0,2]),packed)
repeat,_=expert.transition(x,torch.tensor([0,2]),packed)
assert torch.equal(first,repeat)
optimizer=torch.optim.AdamW(expert.parameters(),lr=.001)
loss=torch.distributions.Normal(first,std).log_prob(first.detach()+.1).mean().neg()
loss.backward()
assert model.encoder.weight.grad is None
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in expert.parameters())
optimizer.step()
assert not torch.equal(before,model.action_model.action_decoder.layer2.weight)
assert torch.equal(frozen,model.encoder.weight)
assert all(torch.equal(cached[k],v) for k,v in packed.items())
expert.assert_frozen_encoder()
expert2.load_state_dict(expert.state_dict(),strict=True)
torch.testing.assert_close(expert.transition(x,torch.tensor([1,2]),packed)[0], expert2.transition(x,torch.tensor([1,2]),packed)[0],rtol=0,atol=0)
for value in (0,-1,float('nan'),float('inf'),True):
    try: VLAJEPAFlowExpert(copy.deepcopy(model),config={'transition_std':value})
    except ValueError: pass
    else: raise AssertionError('accepted invalid stochastic standard deviation')
try: expert.transition(x,torch.tensor([0,3]),packed)
except ValueError: pass
else: raise AssertionError('accepted invalid chain index')
model.encoder.weight.requires_grad_(True)
try: expert.assert_frozen_encoder()
except RuntimeError: pass
else: raise AssertionError('accepted trainable encoder')
'''
    run_native(code, ROOT / "src/pressb/model_sources/vlajepa", tmp_path, dimension, state_dimension)


def test_wall_native_moe_gradients_padding_cached_context_and_weight_transfer(tmp_path):
    code = r'''
import copy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from pressb.models.flow_experts import WallOSSFlowExpert
from pressb.runtime.denoising import GaussianChainPolicy
from wall_x.model.flow_context import split_diffusion_context
from wall_x.model.qwen2_5_based.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
torch.set_num_threads(1)
torch.manual_seed(91)
cfg=Qwen2_5_VLConfig(vocab_size=32,hidden_size=32,intermediate_size=48,
    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
    num_experts=2,mlp_moe=True,attention_dropout=.1,
    experts=[{'hidden_size':32,'intermediate_size':48,'hidden_act':'silu'}]*2,
    rope_scaling={'type':'mrope','mrope_section':[1,1,2]},
    dof_config={'joint':20},agent_pos_config={'joint':20},noise_scheduler={},
    image_token_id=5,video_token_id=6,vision_start_token_id=4,
    vision_config={'depth':1,'hidden_size':16,'intermediate_size':32,'num_heads':4,
    'in_channels':3,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':1,
    'out_hidden_size':32,'window_size':4,'fullatt_block_indexes':[0]})
cfg._attn_implementation='eager'
processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda x:{'<|action|>':7,'<|propri|>':8}[x]))
original_to=torch.Tensor.to
def cpu_beta(t,*args,**kwargs):
    return t if args==('cuda',) else original_to(t,*args,**kwargs)
stats={'tiny':{'joint':{'min':[2.]*20,'delta':[4.]*20}}}
with patch('torch.Tensor.to',cpu_beta),patch('wall_x.model.action_head.action_statistic_dof',stats):
    model=Qwen2_5_VLMoEForAction(cfg,processor=processor)
model2=copy.deepcopy(model)
def backend(m):
    return SimpleNamespace(policy=SimpleNamespace(model=m),horizon=2,noise_dim=20,dimension=7,
                           dataset_name='tiny',num_inference_timesteps=3,action_scale=[.1]*6+[.01])
expert=WallOSSFlowExpert(backend(model),config={'transition_std':.03})
expert2=WallOSSFlowExpert(backend(model2),config={'transition_std':.03})
names=[n for n,p in model.named_parameters() if p.requires_grad]
assert names and all('.moe.experts.1.' in n or n.startswith(('action_preprocessor.w1.','action_preprocessor.w2.','action_preprocessor.w3.','action_preprocessor.action_proj_back.')) for n in names)
assert all(name.startswith('expert_modules.') for name,_ in expert.named_parameters())
assert not any('normalizer' in k or 'propri' in k for k in expert.state_dict())
ids=torch.tensor([[0,0,4,8,7,7],[2,3,4,8,7,7]])
mask=ids.ne(0).long()
positions=(mask.cumsum(-1)-1).masked_fill(~mask.bool(),1)
with torch.inference_mode():
    inputs=dict(pred_horizon=2,action_dim=20,num_inference_timesteps=3,input_ids=ids,
                inputs_embeds=model.model.embed_tokens(ids),attention_mask=mask,
                position_ids=positions[None].expand(3,-1,-1),moe_token_types=ids.eq(7).long(),
                dof_mask=torch.ones(2,2,20),dataset_names=['tiny','tiny'])
    rows=split_diffusion_context(dict(inputs=inputs,feature=torch.zeros(2,32)))
    contexts=[SimpleNamespace(conditioning=row) for row in rows]
    packed=expert.pack_contexts(contexts)
single=[expert.pack_contexts([context]) for context in contexts]
collated=expert.collate_conditions(single)
assert all(torch.equal(collated[k],v) for k,v in packed.items())
assert all(not v.is_inference() and not v.requires_grad for v in packed.values())
assert packed['position_ids'].shape==(2,3,6)
def permute(hidden,indices):
    order=indices.argsort(stable=True)
    return hidden[order],order
def unpermute(hidden,order,probabilities):
    return torch.zeros_like(hidden).index_copy(0,order,hidden)
# CUDA-only permutation uses the exact equivalent differentiable CPU indexing.
ops='wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act.ops'
x=torch.randn(2,2,20)
with patch(ops+'.permute',side_effect=permute),patch(ops+'.unpermute',side_effect=unpermute):
    expected=model.decode_diffusion_context({'inputs':inputs},initial_noise=x)['predict_action']
    actual=x.clone()
    for k in range(3):
        actual,std=expert.transition(actual,torch.full((2,),k),packed)
        assert torch.all(std==.03)
    torch.testing.assert_close(expert.actions_from_latent(actual),expected[:,:,:7]*torch.tensor([.1]*6+[.01]),rtol=2e-6,atol=2e-6)
    chain_policy=GaussianChainPolicy(expert)
    sample=chain_policy.sample(packed,[11,19],initial_noise=x)
    assert sample['chain'].shape == (2,4,2,20) and sample['actions'].shape==(2,2,7)
    for k in range(3):
        score=chain_policy.log_prob(packed,sample['chain'][:,k],sample['chain'][:,k+1],k)
        torch.testing.assert_close(score,sample['logprobs'][:,k],atol=0,rtol=0)
    batch_mean,_=expert.transition(x,torch.tensor([1,2]),packed)
    for i in range(2):
        one,_=expert.transition(x[i:i+1],torch.tensor([i+1]),single[i])
        torch.testing.assert_close(one,batch_mean[i:i+1],rtol=0,atol=0)
    cached={k:v.clone() for k,v in packed.items()}
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(expert.parameters(),lr=.001)
    mean,std=expert.transition(x,torch.tensor([1,2]),packed)
    loss=-torch.distributions.Normal(mean,std).log_prob(mean.detach()+.08).mean()
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if n not in names)
    assert all(torch.isfinite(p.grad).all() for p in expert.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.model.layers[0].moe.experts[1].parameters())
    optimizer.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    assert any('.moe.experts.1.' in n for n in changed) and 'action_preprocessor.action_proj_back.weight' in changed
    assert all(n in names for n in changed)
    assert all(torch.equal(cached[k],v) for k,v in packed.items())
    expert.assert_frozen_encoder()
    expert2.load_state_dict(expert.state_dict(),strict=True)
    torch.testing.assert_close(expert.transition(x,torch.tensor([0,2]),packed)[0],expert2.transition(x,torch.tensor([0,2]),packed)[0],rtol=0,atol=0)
    damaged=dict(packed,moe_token_types=packed['moe_token_types'].clone())
    damaged['moe_token_types'][0,2]=1
    try: expert.transition(x,torch.tensor([0,2]),damaged)
    except ValueError: pass
    else: raise AssertionError('accepted action route outside action tokens')
    fixed=WallOSSFlowExpert(backend(copy.deepcopy(model)),config={
        'transition_std':.03,'transition_microbatch_size':8,'transition_context_length':8})
    assert fixed.describe()['transition_context_length']==8
    conditions=fixed.collate_conditions([single[i%2] for i in range(11)])
    fixed_policy=GaussianChainPolicy(fixed)
    with patch.object(fixed,'_transition_row',wraps=fixed._transition_row) as calls:
        draws=fixed_policy.sample(conditions,list(range(11)))
        assert calls.call_count==6
        assert all(call.args[0].shape[0]==8 and call.args[2]['input_ids'].shape==(8,8) for call in calls.call_args_list)
    # Variable sequence lengths, independent transport padding, permutations,
    # one-row requests and a three-row tail all use the same kernel shapes.
    order=torch.tensor([10,3,0,5,2,7,8,1,6,9,4])
    permuted={key:value[order] for key,value in conditions.items()}
    for k in range(3):
        score=fixed_policy.log_prob(permuted,draws['chain'][order,k],draws['chain'][order,k+1],k)
        torch.testing.assert_close(score,draws['logprobs'][order,k],rtol=0,atol=0)
        one=fixed_policy.log_prob(single[0],draws['chain'][:1,k],draws['chain'][:1,k+1],k)
        torch.testing.assert_close(one,draws['logprobs'][:1,k],rtol=0,atol=0)
    steps=torch.tensor([2,0,1,1,2,0,2,1,0,1,2])
    score=fixed_policy.log_prob(permuted,draws['chain'][order,steps],draws['chain'][order,steps+1],steps)
    torch.testing.assert_close(score,draws['logprobs'][order,steps],rtol=0,atol=0)
    (-score.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in fixed.parameters())
    assert all(p.grad is None for n,p in fixed._source_model.named_parameters() if not p.requires_grad)
    fixed.transition_context_length=3
    try: fixed.transition(x,torch.tensor([0,2]),packed)
    except ValueError as exc: assert 'exceeding' in str(exc)
    else: raise AssertionError('silently truncated a context')
    for bad in ({'transition_microbatch_size':8},{'transition_microbatch_size':2},
                {'transition_microbatch_size':True},{'transition_microbatch_size':1,'transition_context_length':8},
                {'transition_microbatch_size':8,'transition_context_length':0},
                {'transition_microbatch_size':8,'transition_context_length':True}):
        try: WallOSSFlowExpert(backend(None),config=bad)
        except ValueError: pass
        else: raise AssertionError('accepted invalid fixed-batch configuration')
'''
    run_native(code, ROOT / "src/pressb/model_sources/walloss", tmp_path)
