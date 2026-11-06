"""Physical contact and termination policy shared by collection and runtime."""
from copy import deepcopy


FINGER_PAIR = ('/World/Piper/link7', '/World/Piper/link8')
DUAL_COLLISION_POLICY = dict(version='table_buttons_dual_contact_v1',
    allowed_task_contact_pairs=[list(FINGER_PAIR)],
    task_success='physical_target_press_without_wrong_or_blocking_contact',
    strict_collector_success='task_success_and_no_original_unexpected_contact')
STRICT_COLLISION_POLICY = dict(version='collector_strict_v1', allowed_task_contact_pairs=[],
    task_success='physical_target_press_without_any_original_unexpected_contact',
    strict_collector_success='identical_to_task_success')



def collision_settings(requested=None):
    """Only an explicit frozen plan can permit normal inter-finger closure."""
    if requested is None or requested == STRICT_COLLISION_POLICY:
        return deepcopy(STRICT_COLLISION_POLICY)
    if requested != DUAL_COLLISION_POLICY:
        raise ValueError('Unknown collision policy; only the exact finger-pair exception is supported')
    return deepcopy(DUAL_COLLISION_POLICY)


def is_allowed_finger_closure(actors, collision_policy):
    return bool(collision_policy == DUAL_COLLISION_POLICY and len(actors) == 2
                and set(actors) == set(FINGER_PAIR))


def strict_outcome(result, *, wrong_press=None, wrong_contact=False, unexpected_contacts=(), model_error=None):
    """Apply collector-equivalent failures even after the success latch fired."""
    if model_error is not None:
        return False, 'invalid_policy_action'
    if unexpected_contacts:
        return False, 'unexpected_contact'
    if wrong_contact:
        return False, 'non_target_finger_contact'
    if wrong_press is not None:
        return False, 'wrong_button'
    if result and result['success']:
        return True, 'success'
    return False, 'timeout'
