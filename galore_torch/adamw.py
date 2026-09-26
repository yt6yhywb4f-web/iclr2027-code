# copy dependencies from transformers/optimization.py
import csv
import json
import math
import os
import warnings
from typing import Callable, Iterable, Tuple

import torch

from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version

from .adamw_math import advance_moments_, apply_adaptive_update_and_weight_decay_
from .galore_projector import GaLoreProjector
from .galore_projector_tensor import GaLoreProjectorTensor


def _rank_zero():
    return int(os.environ.get("RANK", "0")) == 0


def _parameter_identity(group, parameter):
    parameter_id = id(parameter)
    return (
        group.get("param_name_by_id", {}).get(parameter_id),
        group.get("param_index_by_id", {}).get(parameter_id),
    )


def _append_csv_row(path, fieldnames, row):
    if not path or not _rank_zero():
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not needs_header:
        with open(path, newline="") as existing:
            if next(csv.reader(existing), []) != list(fieldnames):
                raise ValueError(f"CSV header mismatch at {path}; use a new output file")
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _apply_v_gamma_(exp_avg_sq, gamma):
    if exp_avg_sq is None:
        return False
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"refresh_v_gamma must lie in [0, 1], got {gamma}")
    if gamma == 1.0:
        return False
    if gamma == 0.0:
        exp_avg_sq.zero_()
    else:
        exp_avg_sq.mul_(gamma)
    return True


def _apply_refresh_intervention_(group, state, parameter, refresh_details):
    if refresh_details is None or refresh_details.get("is_initial", False):
        return
    gamma = group.get("refresh_v_gamma")
    if gamma is not None:
        _apply_v_gamma_(state["exp_avg_sq"], gamma)
        if _rank_zero():
            parameter_name, _ = _parameter_identity(group, parameter)
            print(
                f"Refresh v gamma | training_step={state['step']} "
                f"parameter_name={parameter_name} gamma={float(gamma)}"
            )
        return
    ablation = group.get("optimizer_state_ablation", "none")
    if ablation in {"reset_m", "reset_all"}:
        state["exp_avg"].zero_()
    if ablation in {
        "reset_v",
        "reset_all",
        "reset_v_reset_bc",
        "reset_v_local_v_age",
    }:
        state["exp_avg_sq"].zero_()
    if ablation == "reset_v_reset_bc":
        # This reproduces the historical counter-reset control: one shared
        # correction age is restarted while the global refresh step is kept.
        state["bias_correction_step"] = 0
    if ablation == "reset_v_local_v_age":
        # Restart only the age of the reconstructed second moment. The
        # optimizer/global age and the mature first-moment correction remain.
        state["v_age"] = 0


def _log_refresh_event(group, state, parameter, refresh_details):
    path = group.get("refresh_events_csv")
    if not path or refresh_details is None:
        return
    parameter_name, parameter_index = _parameter_identity(group, parameter)
    fieldnames = [
        "training_step",
        "parameter_name",
        "parameter_index",
        "projection_orientation",
        "refresh_index",
        "is_initial",
        "optimizer_state_ablation",
        "refresh_v_gamma",
    ]
    _append_csv_row(path, fieldnames, {
        "training_step": int(state["step"]),
        "parameter_name": parameter_name,
        "parameter_index": parameter_index,
        "projection_orientation": refresh_details["type"],
        "refresh_index": refresh_details["refresh_index"],
        "is_initial": refresh_details["is_initial"],
        "optimizer_state_ablation": group.get("optimizer_state_ablation", "none"),
        "refresh_v_gamma": group.get("refresh_v_gamma"),
    })


def _load_refresh_lr_schedule(group):
    cached = group.get("_resolved_refresh_lr_schedule")
    if cached is not None:
        return cached
    schedule_path = group.get("refresh_lr_schedule_json")
    if schedule_path:
        with open(schedule_path) as handle:
            payload = json.load(handle)
        schedule_kind = group.get("refresh_lr_schedule_kind", "spike")
        field = f"{schedule_kind}_multiplier"
        entries = sorted(payload["offsets"], key=lambda entry: int(entry["offset"]))
        schedule = tuple(float(entry[field]) for entry in entries)
    elif group.get("refresh_lr_multipliers"):
        schedule = tuple(
            float(value.strip())
            for value in str(group["refresh_lr_multipliers"]).split(",")
            if value.strip()
        )
    else:
        window = int(group.get("refresh_lr_window", 0))
        schedule = (float(group.get("refresh_lr_multiplier", 1.0)),) * window
    if any(not math.isfinite(value) or value <= 0.0 for value in schedule):
        raise ValueError("Refresh LR multipliers must be finite and positive")
    group["_resolved_refresh_lr_schedule"] = schedule
    return schedule


def _activate_refresh_lr_control(group, state, parameter, refresh_details):
    if (
        refresh_details is not None
        and not refresh_details.get("is_initial", False)
        and _load_refresh_lr_schedule(group)
    ):
        state["_refresh_lr_base_step"] = int(state["step"]) + 1


def _refresh_lr_multiplier(group, state, parameter):
    schedule = _load_refresh_lr_schedule(group)
    base_step = state.get("_refresh_lr_base_step")
    if not schedule or base_step is None:
        return 1.0, None, False
    offset = int(state["step"]) - base_step
    if offset < 0 or offset >= len(schedule):
        return 1.0, offset, False
    return schedule[offset], offset, True


def _log_refresh_lr(group, state, multiplier, offset, active, base_step_size):
    path = group.get("refresh_lr_control_csv")
    if not path:
        return
    token = (int(state["step"]), multiplier)
    if group.get("_last_refresh_lr_log") == token:
        return
    group["_last_refresh_lr_log"] = token
    fieldnames = [
        "training_step",
        "offset",
        "active",
        "scheduled_lr",
        "effective_lr",
        "bias_corrected_step_size",
        "effective_bias_corrected_step_size",
        "multiplier",
    ]
    _append_csv_row(path, fieldnames, {
        "training_step": int(state["step"]),
        "offset": offset,
        "active": active,
        "scheduled_lr": float(group["lr"]),
        "effective_lr": float(group["lr"]) * multiplier,
        "bias_corrected_step_size": base_step_size,
        "effective_bias_corrected_step_size": base_step_size * multiplier,
        "multiplier": multiplier,
    })


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated; use torch.optim.AdamW when GaLore is not required.",
                FutureWarning,
            )
        require_version("torch>=1.5.0")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.rebind_parameter_metadata()

    def rebind_parameter_metadata(self):
        """Bind serialized ordered names to the current parameter objects."""
        for group in self.param_groups:
            names = group.get("_parameter_names")
            indices = group.get("_parameter_indices")
            if names is None:
                names = [group.get("param_name_by_id", {}).get(id(p)) for p in group["params"]]
            if indices is None:
                indices = [group.get("param_index_by_id", {}).get(id(p)) for p in group["params"]]
            if len(names) != len(group["params"]) or len(indices) != len(group["params"]):
                raise ValueError("Saved parameter metadata does not match optimizer group size")
            group["_parameter_names"] = list(names)
            group["_parameter_indices"] = list(indices)
            group["param_name_by_id"] = {id(p): n for p, n in zip(group["params"], names) if n is not None}
            group["param_index_by_id"] = {id(p): n for p, n in zip(group["params"], indices) if n is not None}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.rebind_parameter_metadata()

    def load_state_dict(self, state_dict):
        # Older checkpoints contain object-id maps only. For these, use the
        # stable metadata supplied by the newly constructed model/optimizer.
        fallback = [(g.get("_parameter_names"), g.get("_parameter_indices"))
                    for g in self.param_groups]
        runtime_outputs = [{key: value for key, value in g.items()
                            if key in {"refresh_events_csv", "refresh_lr_control_csv", "projector_save_dir", "freeze_projector_after_initial"}}
                           for g in self.param_groups]
        result = super().load_state_dict(state_dict)
        for group, saved_group, (names, indices) in zip(
                self.param_groups, state_dict["param_groups"], fallback):
            if names is not None and any(name is not None for name in names):
                group["_parameter_names"] = names
                group["_parameter_indices"] = indices
        for group, outputs in zip(self.param_groups, runtime_outputs):
            group.update(outputs)
            for parameter in group["params"]:
                projector = self.state[parameter].get("projector")
                if projector is not None:
                    for name in ("projector_save_dir",):
                        if name in outputs:
                            setattr(projector, name, outputs[name])
        self.rebind_parameter_metadata()
        return result

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[parameter]
                state.setdefault("step", 0)
                state.setdefault("bias_correction_step", state["step"])
                if group.get("optimizer_state_ablation") == "reset_v_local_v_age":
                    # Existing checkpoints predate v_age. Before the first
                    # reset, v has the same age as the optimizer.
                    state.setdefault("v_age", state["step"])
                group.setdefault("dim", 2)

                projector = None
                if "rank" in group:
                    if "projector" not in state:
                        parameter_name, parameter_index = _parameter_identity(group, parameter)
                        if group["dim"] <= 2:
                            state["projector"] = GaLoreProjector(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                                projector_index=parameter_index,
                                projector_name=parameter_name,
                                projector_save_dir=group.get("projector_save_dir"),
                                projector_load_dir=group.get("projector_load_dir"),
                            )
                        else:
                            state["projector"] = GaLoreProjectorTensor(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                            )
                    projector = state["projector"]
                    suppress_scheduled_refresh = (
                        bool(group.get("freeze_projector_after_initial", False))
                        and projector.ortho_matrix is not None
                    )
                    gradient = projector.project(
                        gradient,
                        state["step"],
                        suppress_scheduled_refresh=suppress_scheduled_refresh,
                    )

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)

                refresh_details = getattr(projector, "last_refresh_details", None)
                _log_refresh_event(group, state, parameter, refresh_details)
                # This is the single intervention point shared by historical
                # reset_v and refresh_v_gamma, before current-gradient moments.
                _apply_refresh_intervention_(group, state, parameter, refresh_details)
                _activate_refresh_lr_control(group, state, parameter, refresh_details)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                state["bias_correction_step"] += 1
                if group.get("optimizer_state_ablation") == "reset_v_local_v_age":
                    state["v_age"] += 1
                advance_moments_(exp_avg, exp_avg_sq, gradient, beta1, beta2)
                denominator = exp_avg_sq.sqrt().add_(group["eps"])

                base_step_size = group["lr"]
                if group["correct_bias"]:
                    m_correction_age = state["bias_correction_step"]
                    v_correction_age = (
                        state["v_age"]
                        if group.get("optimizer_state_ablation") == "reset_v_local_v_age"
                        else m_correction_age
                    )
                    bias_correction1 = 1.0 - beta1 ** m_correction_age
                    bias_correction2 = 1.0 - beta2 ** v_correction_age
                    base_step_size *= math.sqrt(bias_correction2) / bias_correction1
                normalized_update = exp_avg / denominator
                if projector is not None:
                    normalized_update = projector.project_back(normalized_update)

                multiplier, offset, active = _refresh_lr_multiplier(group, state, parameter)
                _log_refresh_lr(group, state, multiplier, offset, active, base_step_size)
                apply_adaptive_update_and_weight_decay_(
                    parameter,
                    normalized_update,
                    base_step_size,
                    multiplier,
                    group["lr"],
                    group["weight_decay"],
                )
        return loss
