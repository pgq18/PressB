import json
import os
import torch
import torch_npu
import torch.distributed as dist
from datetime import timedelta

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(local_rank)
dist.init_process_group('hccl', timeout=timedelta(minutes=3))
device = torch.device(f'npu:{local_rank}')
value = torch.tensor([rank + 1.], device=device)
dist.all_reduce(value)
assert float(value) == dist.get_world_size() * (dist.get_world_size() + 1) / 2
x = torch.randn(128,128,device=device,dtype=torch.bfloat16,requires_grad=True)
loss=(x@x).float().square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
torch.npu.synchronize()
print(json.dumps(dict(rank=rank,device=str(device),sum=float(value),bf16_backward=True)),flush=True)
dist.barrier()
dist.destroy_process_group()
