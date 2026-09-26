import torch

from galore_torch.adamw import _apply_refresh_intervention_


def _state():
    return {
        "step": 200,
        "bias_correction_step": 200,
        "exp_avg": torch.tensor([1.0, 2.0]),
        "exp_avg_sq": torch.tensor([3.0, 4.0]),
    }


def _apply(condition):
    state = _state()
    group = {"optimizer_state_ablation": condition, "refresh_v_gamma": None}
    refresh = {"is_initial": False}
    _apply_refresh_intervention_(group, state, object(), refresh)
    return state


def test_carry_preserves_both_moments_and_age():
    state = _apply("none")
    assert torch.equal(state["exp_avg"], torch.tensor([1.0, 2.0]))
    assert torch.equal(state["exp_avg_sq"], torch.tensor([3.0, 4.0]))
    assert state["bias_correction_step"] == 200


def test_reset_m_only_zeros_first_moment():
    state = _apply("reset_m")
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.equal(state["exp_avg_sq"], torch.tensor([3.0, 4.0]))
    assert state["bias_correction_step"] == 200


def test_reset_v_only_zeros_second_moment():
    state = _apply("reset_v")
    assert torch.equal(state["exp_avg"], torch.tensor([1.0, 2.0]))
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0
    assert state["bias_correction_step"] == 200


def test_reset_all_zeros_both_moments_without_resetting_global_age():
    state = _apply("reset_all")
    assert torch.count_nonzero(state["exp_avg"]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0
    assert state["bias_correction_step"] == 200


def test_local_v_age_preserves_m_and_restarts_only_v_age():
    state = _apply("reset_v_local_v_age")
    assert torch.equal(state["exp_avg"], torch.tensor([1.0, 2.0]))
    assert torch.count_nonzero(state["exp_avg_sq"]) == 0
    assert state["bias_correction_step"] == 200
    assert state["v_age"] == 0

