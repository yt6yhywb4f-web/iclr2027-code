import copy
import importlib.machinery
import math
import sys
import types
import unittest

import torch


try:
    import tensorly  # noqa: F401
except ImportError:
    tensorly_module = types.ModuleType("tensorly")
    decomposition_module = types.ModuleType("tensorly.decomposition")
    decomposition_module.tucker = lambda *args, **kwargs: None
    tensorly_module.decomposition = decomposition_module
    tensorly_module.tenalg = types.SimpleNamespace()
    sys.modules["tensorly"] = tensorly_module
    sys.modules["tensorly.decomposition"] = decomposition_module

try:
    import bitsandbytes  # noqa: F401
except ImportError:
    bitsandbytes_module = types.ModuleType("bitsandbytes")
    bitsandbytes_module.__spec__ = importlib.machinery.ModuleSpec(
        "bitsandbytes", loader=None
    )
    optim_module = types.ModuleType("bitsandbytes.optim")
    optim_module.__spec__ = importlib.machinery.ModuleSpec(
        "bitsandbytes.optim", loader=None
    )
    optimizer_module = types.ModuleType("bitsandbytes.optim.optimizer")
    optimizer_module.__spec__ = importlib.machinery.ModuleSpec(
        "bitsandbytes.optim.optimizer", loader=None
    )
    optimizer_module.Optimizer2State = object
    optim_module.optimizer = optimizer_module
    bitsandbytes_module.optim = optim_module
    sys.modules["bitsandbytes"] = bitsandbytes_module
    sys.modules["bitsandbytes.optim"] = optim_module
    sys.modules["bitsandbytes.optim.optimizer"] = optimizer_module

from galore_torch.adamw import AdamW
from galore_torch.galore_projector import GaLoreProjector


def _group(parameter, mode="none", proj_type="right"):
    return {
        "params": [parameter],
        "rank": 1,
        "update_proj_gap": 2,
        "scale": 1.0,
        "proj_type": proj_type,
        "optimizer_state_ablation": mode,
        "param_name_by_id": {id(parameter): "weight"},
        "param_index_by_id": {id(parameter): 0},
    }


def _optimizer(parameter, mode="none", proj_type="right"):
    return AdamW(
        [_group(parameter, mode=mode, proj_type=proj_type)],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-6,
        no_deprecation_warning=True,
    )


def _step(parameter, optimizer, gradient):
    parameter.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad()


class LocalVAgeOptimizerTests(unittest.TestCase):
    def test_projector_checksum_supports_left_stride_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            basis = torch.arange(4.0, dtype=dtype).reshape(1, 4).t()
            checksum = GaLoreProjector._basis_sha256(basis)
            self.assertEqual(len(checksum), 64)

    def _warm_pair(self, proj_type="right"):
        initial = torch.arange(1.0, 17.0).reshape(4, 4) / 16.0
        plain_parameter = torch.nn.Parameter(initial.clone())
        local_parameter = torch.nn.Parameter(initial.clone())
        plain = _optimizer(plain_parameter, proj_type=proj_type)
        local = _optimizer(
            local_parameter,
            mode="reset_v_local_v_age",
            proj_type=proj_type,
        )
        for scale in (0.2, -0.4):
            gradient = torch.arange(1.0, 17.0).reshape(4, 4) * scale
            _step(plain_parameter, plain, gradient)
            _step(local_parameter, local, gradient)
        return plain_parameter, plain, local_parameter, local

    def test_disabled_path_is_bitwise_unchanged_before_first_reset(self):
        plain_parameter, plain, local_parameter, local = self._warm_pair()
        torch.testing.assert_close(plain_parameter, local_parameter, rtol=0, atol=0)
        plain_state = plain.state[plain_parameter]
        local_state = local.state[local_parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(plain_state[key], local_state[key], rtol=0, atol=0)
        self.assertNotIn("v_age", plain_state)
        self.assertEqual(local_state["v_age"], local_state["step"])

    def test_refresh_resets_only_v_age_and_v(self):
        _, plain, local_parameter, local = self._warm_pair()
        local_state = local.state[local_parameter]
        m_before = local_state["exp_avg"].clone()
        global_step_before = local_state["step"]
        correction_step_before = local_state["bias_correction_step"]
        gradient = torch.flip(torch.arange(1.0, 17.0).reshape(4, 4), dims=(1,))
        _step(local_parameter, local, gradient)

        self.assertEqual(local_state["step"], global_step_before + 1)
        self.assertEqual(
            local_state["bias_correction_step"], correction_step_before + 1
        )
        self.assertEqual(local_state["v_age"], 1)
        projected = local_state["projector"].project_with_basis(
            gradient,
            local_state["projector"].ortho_matrix,
            local_state["projector"].proj_type,
        )
        torch.testing.assert_close(
            local_state["exp_avg"],
            0.9 * m_before + 0.1 * projected,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            local_state["exp_avg_sq"],
            0.001 * projected.square(),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_local_age_uses_global_m_and_local_v_corrections(self):
        _, _, parameter, optimizer = self._warm_pair()
        before = parameter.detach().clone()
        _step(parameter, optimizer, torch.eye(4))
        state = optimizer.state[parameter]
        self.assertEqual(state["bias_correction_step"], 3)
        self.assertEqual(state["v_age"], 1)
        direction = state["projector"].project_back(
            state["exp_avg"] / state["exp_avg_sq"].sqrt().add_(1e-6)
        )
        # Compare the executed update, with global first-moment correction and
        # local second-moment correction, directly against a native addition.
        step_size = 0.01 * math.sqrt(1.0 - 0.999) / (1.0 - 0.9 ** 3)
        expected = before.clone().add_(direction, alpha=-step_size)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        wrong_global_v = before.clone().add_(
            direction, alpha=-0.01 * math.sqrt(1.0 - 0.999 ** 3) / (1.0 - 0.9 ** 3)
        )
        wrong_local_m = before.clone().add_(
            direction, alpha=-0.01 * math.sqrt(1.0 - 0.999) / (1.0 - 0.9)
        )
        self.assertFalse(torch.equal(parameter, wrong_global_v))
        self.assertFalse(torch.equal(parameter, wrong_local_m))

    def test_v_age_serializes_and_restores(self):
        _, _, parameter, optimizer = self._warm_pair()
        gradient = torch.eye(4)
        _step(parameter, optimizer, gradient)
        saved = copy.deepcopy(optimizer.state_dict())

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = _optimizer(restored_parameter, mode="reset_v_local_v_age")
        restored.load_state_dict(saved)
        state = restored.state[restored_parameter]
        self.assertEqual(state["v_age"], 1)
        self.assertEqual(state["step"], 3)
        self.assertEqual(state["bias_correction_step"], 3)

    def test_left_and_right_paths_share_counter_semantics(self):
        for proj_type in ("left", "right"):
            with self.subTest(proj_type=proj_type):
                _, _, parameter, optimizer = self._warm_pair(proj_type=proj_type)
                _step(parameter, optimizer, torch.eye(4))
                state = optimizer.state[parameter]
                self.assertEqual(state["step"], 3)
                self.assertEqual(state["bias_correction_step"], 3)
                self.assertEqual(state["v_age"], 1)

    def test_deepcopied_branches_do_not_share_v_age_or_moments(self):
        _, _, parameter, optimizer = self._warm_pair()
        branch_parameter, branch_optimizer = copy.deepcopy((parameter, optimizer))
        source_state = optimizer.state[parameter]
        branch_state = branch_optimizer.state[branch_parameter]
        branch_state["v_age"] = 99
        branch_state["exp_avg"].zero_()
        self.assertNotEqual(source_state["v_age"], branch_state["v_age"])
        self.assertFalse(torch.equal(source_state["exp_avg"], branch_state["exp_avg"]))



class LocalVAgeAnalyticTests(unittest.TestCase):
    def test_stationary_scalar_prediction(self):
        beta1 = 0.9
        beta2 = 0.999
        global_age = 5000
        mature_m = 1.0
        for local_t in (1, 2, 5, 10, 20):
            v = 1.0 - beta2 ** local_t
            m = beta1 ** local_t * mature_m + (1.0 - beta1 ** local_t)
            global_update = (m / (1.0 - beta1 ** (global_age + local_t))) / math.sqrt(
                v / (1.0 - beta2 ** (global_age + local_t))
            )
            local_update = (m / (1.0 - beta1 ** (global_age + local_t))) / math.sqrt(
                v / (1.0 - beta2 ** local_t)
            )
            predicted = math.sqrt(
                (1.0 - beta2 ** (global_age + local_t))
                / (1.0 - beta2 ** local_t)
            )
            self.assertAlmostEqual(global_update / local_update, predicted, places=12)
            self.assertAlmostEqual(local_update, 1.0, places=12)

            global_counter_reset_update = (
                m / (1.0 - beta1 ** local_t)
            ) / math.sqrt(v / (1.0 - beta2 ** local_t))
            self.assertGreater(global_counter_reset_update, local_update)


if __name__ == "__main__":
    unittest.main()
