import numpy as np
import pytest
import torch

from scripts.diagnose_search_leaf_values import leaf_path, leaf_search, same_search
from src.muzero.mcts import MCTS
from src.muzero.node import Node


class Net:
    def initial_inference(self, obs):
        return torch.zeros(1,2,2,2), torch.zeros(1,4), torch.zeros(1)

    def recurrent_inference(self, hidden, actions):
        n=len(actions)
        return hidden+.01, torch.full((n,),.2), torch.zeros(n,4), torch.zeros(n)


def lookup(path):
    # Deliberately consume numpy RNG to test isolation of observation queries.
    np.random.random(7)
    return {'value':0., 'terminal':False, 'terminal_depth':None,
            'observation_sha256':'test'}


@pytest.mark.parametrize('batch',[1,4])
@pytest.mark.parametrize('temperature',[0.,.25])
def test_observe_only_preserves_exact_search_and_rng(batch,temperature):
    search=MCTS(.99,50,root_exploration_eps=0.,device='cpu',leaf_batch=batch)
    obs=np.zeros((4,8,8),dtype=np.float32)
    np.random.seed(91)
    expected=search.run(obs,Net(),temperature=temperature)
    state=np.random.get_state()
    np.random.seed(91)
    actual,diagnostic=leaf_search(search,obs,Net(),temperature,lookup,False)
    same_search(expected,actual)
    assert all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state()))
    assert len(diagnostic['leaf_backups'])==50
    assert all(e['depth']==len(e['path']) and e['depth']>=1 for e in diagnostic['leaf_backups'])


def test_substitution_changes_backup_values_and_can_change_selected_action():
    search=MCTS(.99,50,root_exploration_eps=0.,device='cpu',leaf_batch=4)
    obs=np.zeros((4,8,8),dtype=np.float32)
    np.random.seed(45)
    baseline,_=leaf_search(search,obs,Net(),0.,lookup,False)
    preferred=(baseline[0]+1)%4
    def evaluator(path):
        return {**lookup(path),'value':100. if path[0]==preferred else -100.}
    np.random.seed(45)
    result,diagnostic=leaf_search(search,obs,Net(),0.,evaluator,True)
    assert result[0]==preferred
    for event in diagnostic['leaf_backups']:
        assert event['imagined_value']==0.
        assert event['backed_up_value']==(100. if event['path'][0]==preferred else -100.)
        assert event['predicted_reward']==pytest.approx(.2)
    assert diagnostic['root_prior']==[.25]*4


def test_terminal_leaf_value_zero_without_changing_predicted_reward():
    search=MCTS(.99,8,root_exploration_eps=0.,device='cpu',leaf_batch=4)
    def terminal(path):
        return {**lookup(path),'value':999.,'terminal':True,'terminal_depth':1}
    _,diagnostic=leaf_search(search,np.zeros((4,8,8),dtype=np.float32),Net(),0.,terminal,True)
    assert all(e['backed_up_value']==0. and e['predicted_reward']==pytest.approx(.2)
               for e in diagnostic['leaf_backups'])


def test_paths_keep_order_and_backup_patch_is_restored_after_error():
    root=Node(0.)
    first=Node(.5,3,root)
    last=Node(.5,1,first)
    assert leaf_path(last)==(3,1)
    original=Node.backup
    def error(path):
        raise RuntimeError('failure')
    with pytest.raises(RuntimeError,match='failure'):
        leaf_search(MCTS(.99,4,device='cpu'),np.zeros((4,8,8),dtype=np.float32),Net(),0.,error,True)
    assert Node.backup is original


def test_repeated_worker_replays_restore_original_random_start(monkeypatch):
    from types import SimpleNamespace
    import scripts.diagnose_search_leaf_values as module
    env=SimpleNamespace(rng=np.random.default_rng(123))
    def replay(env, record, trace, step):
        return env.rng.integers(0,31,size=10).tolist()
    monkeypatch.setattr(module,'replay_to_branch',replay)
    record={'env_seed':5100002}
    expected=np.random.default_rng(5100002).integers(0,31,size=10).tolist()
    assert module.replay_original_root(env,record,{},147)==expected
    env.rng.random(100)
    assert module.replay_original_root(env,record,{},147)==expected


def test_search_diagnostics_survive_rollout_length_metadata():
    from scripts.diagnose_search_leaf_values import branch_result
    diagnostics=[{'step':0,'action_changed':True}]
    result=branch_result(7100001,1,diagnostics,{'decisions':123,'completed':True})
    assert result['decisions']==123
    assert result['search_decisions']==diagnostics
    assert result['completed']
