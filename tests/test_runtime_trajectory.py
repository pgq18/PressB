import json
import numpy as np
import pytest
import torch

from pressb.runtime.trainer import GenericTrainer
from pressb.runtime.trajectory import ChunkTrajectoryWriter
from test_rl_nodes_trainer import config, metadata, Inference, Simulation


@pytest.mark.parametrize('images', [True, False])
def test_generic_collection_matches_validated_episode_budget(tmp_path, images):
    torch.set_num_threads(1)
    im, sm = metadata()
    cfg = config(tmp_path/'run', method='base', steps=20)
    cfg['mode'] = 'eval'
    writer = ChunkTrajectoryWriter(tmp_path/'data', include_images=images)
    result = GenericTrainer(cfg, inference_client=Inference(im), simulation_client=Simulation(sm), observer=writer).run()
    manifest = json.loads((tmp_path/'data/manifest.json').read_text())
    assert result['state'] == 'completed'
    assert sum(x['complete'] for x in manifest['episodes'].values()) == cfg['rollout']['eval_episodes']
    assert manifest['transitions'] == result['transitions']
    for episode in manifest['episodes'].values():
        path = tmp_path/'data'/episode['directory']
        initial = json.loads((path/'initial.json').read_text())
        assert ('images' in initial['observation']) == images
        rows = [json.loads(line) for line in (path/'transitions.jsonl').read_text().splitlines()]
        assert len(rows) == episode['transitions']
        assert rows[0]['observation']['state'] == initial['observation']['state']
        for row in rows:
            assert np.shape(row['requested_action_prefix']) == (2, 7)
            assert 0 < row['info']['elapsed_control_steps'] <= 2
        assert rows[-1]['terminated'] or rows[-1]['truncated']
    writer.close({'state': 'ignored_second_close'})
    assert json.loads((tmp_path/'data/manifest.json').read_text()) == manifest


def test_collector_never_overwrites_existing_dataset(tmp_path):
    with pytest.raises(FileExistsError):
        ChunkTrajectoryWriter(tmp_path)
