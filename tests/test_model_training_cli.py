import json
from pathlib import Path
import sys

import pytest

from pressb.training import cli


@pytest.mark.parametrize('family,profile,module', [
    ('walloss', None, None),
    *[('vlajepa', profile, module) for profile,module in cli.VLA_TRAINERS.items()],
])
def test_dry_run_uses_bundled_source_without_launch(tmp_path, monkeypatch, capsys, family, profile, module):
    config=tmp_path/'config.yaml'; config.write_text('seed: 42\n')
    root=tmp_path/'source'; root.mkdir()
    entry=root/(module.replace('.','/')+'.py' if module else 'train_qact.py')
    entry.parent.mkdir(parents=True, exist_ok=True); entry.write_text('raise AssertionError("must not launch")\n')
    monkeypatch.setattr(cli,'builtin_source_root',lambda family:root)
    argv=['--model',family,'--config',str(config),'--dry-run']
    if profile: argv+=['--profile',profile]
    assert cli.main(argv)==0
    result=json.loads(capsys.readouterr().out)
    assert result['source']=='bundled' and result['module']==module
    assert result['launches_training'] is False


def test_wall_dispatches_native_args_and_restores_argv(tmp_path, monkeypatch):
    root=tmp_path/'wall'; root.mkdir(); (root/'train_qact.py').write_text('# entry\n')
    config=tmp_path/'train.yaml'; config.write_text('seed: 42\n')
    seen=[]
    monkeypatch.setattr(cli,'activate_source',lambda family,root: seen.append((family,str(root))))
    monkeypatch.setattr(cli.runpy,'run_path',lambda path,run_name: seen.append((path,run_name,list(sys.argv))))
    argv=sys.argv
    assert cli.main(['--model','walloss','--source-root',str(root),'--config',str(config),'--','--seed','7'])==0
    assert seen[0]==('walloss',str(root))
    assert seen[1][2][1:]==['--config',str(config),'--seed','7']
    assert sys.argv is argv


def test_wall_rejects_vla_profile(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(['--model','walloss','--profile','joint6','--config',str(tmp_path/'x.yaml')])
