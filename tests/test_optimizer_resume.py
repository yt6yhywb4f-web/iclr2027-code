import copy
import io

import pytest
import torch

from galore_torch.adamw import AdamW, _parameter_identity


def optimizer(parameter, mode="none", **extra):
    group = dict(params=[parameter], rank=2, update_proj_gap=2, scale=.25,
                 proj_type="std", optimizer_state_ablation=mode,
                 param_name_by_id={id(parameter): "model.layers.0.self_attn.q_proj.weight"},
                 param_index_by_id={id(parameter): 0}, **extra)
    return AdamW([group], lr=.01, eps=1e-6, no_deprecation_warning=True)


def advance(parameter, opt, gradient):
    parameter.grad = gradient.clone()
    opt.step()
    opt.zero_grad()


@pytest.mark.parametrize("mode", ["none", "reset_m", "reset_v", "reset_all", "reset_v_local_v_age"])
def test_disk_resume_preserves_native_updates_identity_and_local_lr(mode):
    torch.manual_seed(23)
    parameter = torch.nn.Parameter(torch.randn(8, 6).bfloat16())
    opt = optimizer(parameter, mode, refresh_lr_multipliers="0.1,0.2")
    for _ in range(3):
        advance(parameter, opt, torch.randn_like(parameter))
    saved = io.BytesIO()
    torch.save(opt.state_dict(), saved)
    saved.seek(0)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = optimizer(restored_parameter, mode)
    restored.load_state_dict(torch.load(saved, weights_only=False))
    assert _parameter_identity(restored.param_groups[0], restored_parameter) == _parameter_identity(opt.param_groups[0], parameter)
    for _ in range(4):
        gradient = torch.randn_like(parameter)
        advance(parameter, opt, gradient)
        advance(restored_parameter, restored, gradient)
        torch.testing.assert_close(restored_parameter, parameter, atol=0, rtol=0)
        assert restored.state[restored_parameter]["step"] == opt.state[parameter]["step"]


def test_deepcopy_rebinds_metadata_without_sharing_parameters_or_moments():
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    opt = optimizer(parameter, "reset_v_local_v_age")
    advance(parameter, opt, torch.randn_like(parameter))
    copied_parameter, copied = copy.deepcopy((parameter, opt))
    assert _parameter_identity(copied.param_groups[0], copied_parameter) == _parameter_identity(opt.param_groups[0], parameter)
    assert copied_parameter is not parameter
    copied.state[copied_parameter]["exp_avg"].zero_()
    assert torch.count_nonzero(opt.state[parameter]["exp_avg"]) > 0



def test_projector_replay_preserves_training_updates(tmp_path):
    torch.manual_seed(31)
    initial = torch.randn(8, 6).bfloat16()
    gradients = [torch.randn_like(initial) for _ in range(5)]
    source_parameter = torch.nn.Parameter(initial.clone())
    source = optimizer(source_parameter, "reset_v_local_v_age", projector_save_dir=str(tmp_path))
    endpoints = []
    for gradient in gradients:
        advance(source_parameter, source, gradient)
        endpoints.append(source_parameter.detach().clone())
    replay_parameter = torch.nn.Parameter(initial.clone())
    replay = optimizer(replay_parameter, "reset_v_local_v_age", projector_load_dir=str(tmp_path))
    for gradient, expected in zip(gradients, endpoints):
        advance(replay_parameter, replay, gradient)
        torch.testing.assert_close(replay_parameter, expected, rtol=0, atol=0)


def test_freeze_keeps_initial_basis_and_scheduled_local_resets():
    torch.manual_seed(41)
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    opt = optimizer(parameter, "reset_v_local_v_age", freeze_projector_after_initial=True)
    advance(parameter, opt, torch.randn_like(parameter))
    state = opt.state[parameter]
    initial_basis = state["projector"].ortho_matrix.clone()
    for _ in range(4):
        advance(parameter, opt, torch.randn_like(parameter))
        torch.testing.assert_close(state["projector"].ortho_matrix, initial_basis, rtol=0, atol=0)
    assert state["projector"].refresh_count == 1
    assert state["step"] == 5
    assert state["v_age"] == 1
    assert state["projector"].last_refresh_details["refresh_suppressed"] is True


def test_resume_preserves_configured_freeze_when_checkpoint_lacks_new_key():
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.randn(8, 6))
    opt = optimizer(parameter, "reset_v_local_v_age", freeze_projector_after_initial=True)
    for _ in range(3):
        advance(parameter, opt, torch.randn_like(parameter))
    saved = copy.deepcopy(opt.state_dict())
    saved["param_groups"][0].pop("freeze_projector_after_initial")
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = optimizer(restored_parameter, "reset_v_local_v_age", freeze_projector_after_initial=True)
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["freeze_projector_after_initial"] is True
    for _ in range(2):
        gradient = torch.randn_like(parameter)
        advance(parameter, opt, gradient)
        advance(restored_parameter, restored, gradient)
        torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
    assert restored.state[restored_parameter]["projector"].refresh_count == 1
