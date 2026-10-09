"""Check actual CUDA computation on both local GPUs without loading a simulator."""
import importlib.metadata as metadata
import json
from pathlib import Path
import torch
import warp as wp

@wp.kernel
def square_indices(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = float(i * i)

root = Path(__file__).resolve().parents[1]
assert torch.version.cuda == '12.8', torch.version.cuda
assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
wp.init()
results = []
for index in range(torch.cuda.device_count()):
    device = f'cuda:{index}'
    x = torch.arange(32, dtype=torch.float32, device=device)
    expected = float(sum(i * i for i in range(32)))
    actual = float((x * x).sum().cpu())
    assert actual == expected, (index, actual, expected)
    identity = torch.eye(64, device=device)
    product = identity @ identity
    assert bool(torch.equal(product, identity))
    values = wp.zeros(32, dtype=wp.float32, device=device)
    wp.launch(square_indices, dim=32, inputs=[values], device=device)
    wp.synchronize_device(device)
    assert values.numpy().tolist() == [float(i * i) for i in range(32)]
    results.append(dict(index=index, name=torch.cuda.get_device_name(index),
                        capability=list(torch.cuda.get_device_capability(index)),
                        torch_kernel_pass=True, torch_matmul_pass=True, warp_kernel_pass=True))
    del values, x, identity, product
    torch.cuda.empty_cache()
record = dict(torch=metadata.version('torch'), cuda=torch.version.cuda,
              warp=metadata.version('warp-lang'), devices=results)
(root/'logs/gpu-verification-5090.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
