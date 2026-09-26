"""Native moment and parameter updates used by GaLore AdamW."""


def advance_moments_(m, v, gradient, beta1, beta2):
    # Keep this operation order: BF16 out-of-place products round differently.
    m.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)


def apply_adaptive_update_and_weight_decay_(
    parameter, normalized_update, bias_corrected_step_size, multiplier,
    scheduled_lr, weight_decay,
):
    parameter.add_(normalized_update,
                   alpha=-(float(bias_corrected_step_size) * float(multiplier)))
    if weight_decay > 0.0:
        parameter.add_(parameter, alpha=-(float(scheduled_lr) * float(weight_decay)))

