"""Supervised battle losses and held-out temperature scaling."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from labyrinth_nn.data.sample_schema import OUTCOME_WIN
from labyrinth_nn.models.critic import CriticConfig, TwoTowerBattleCritic


def compute_critic_loss(outputs, batch, config: CriticConfig) -> tuple[Tensor, dict[str, Tensor]]:
    target = batch["outcome_probs_target"].float()
    logits = outputs["outcome_logits"].float()
    outcome = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    mean = outputs["damage_mean"].float().reshape(-1)
    log_std = outputs["damage_log_std"].float().reshape(-1)
    damage_target = batch["mean_damage_ratio"].float().reshape_as(mean)
    huber = F.huber_loss(mean, damage_target)
    variance = torch.exp(2 * log_std)
    gaussian = 0.5 * ((damage_target - mean).square() / variance + 2 * log_std).mean()
    damage = huber + gaussian

    time_pred = outputs["win_rem_time"].float().reshape(-1)
    has_time = batch["has_win_target"].bool().reshape(-1)
    if has_time.any():
        time = F.huber_loss(time_pred[has_time], batch["mean_win_rem_time"].float().reshape(-1)[has_time])
    else:
        time = 0.0 * time_pred.sum()

    affinity = outputs["unit_affinity"].float()
    slot_mask = batch["team_slot_mask"].float()
    team_logits = (affinity.gather(1, batch["team_roster_pos"]) * slot_mask).sum(1)
    team_logits = team_logits / slot_mask.sum(1).clamp_min(1)
    roster_mask = batch["roster_mask"].float()
    norm = (affinity.square() * roster_mask).sum() / roster_mask.sum().clamp_min(1)
    auxiliary = F.binary_cross_entropy_with_logits(team_logits, target[:, OUTCOME_WIN]) + 1e-4 * norm
    total = (outcome + config.damage_loss_weight * damage + config.time_loss_weight * time
             + config.affinity_loss_weight * auxiliary)
    return total, {"total_loss": total.detach(), "outcome_loss": outcome.detach(),
                   "damage_loss": damage.detach(), "damage_huber": huber.detach(), "damage_nll": gaussian.detach(),
                   "time_loss": time.detach(), "affinity_loss": auxiliary.detach()}


def batch_to_device(batch, device):
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def fit_temperature(model: TwoTowerBattleCritic, val_loader, device) -> dict[str, float]:
    """Optimize one scalar on frozen logits. Keep the starting temperature if the fit regresses."""
    was_training = model.training
    model.eval()
    logits, targets = [], []
    try:
        with torch.no_grad():
            for raw in val_loader:
                batch = batch_to_device(raw, device)
                target = batch["outcome_probs_target"]
                mask = batch.get("has_outcome_target")
                values = model(batch)["outcome_logits"]
                if mask is not None:
                    values, target = values[mask.reshape(-1)], target[mask.reshape(-1)]
                if values.shape[0]:
                    logits.append(values.detach().to(device="cpu", dtype=torch.float64))
                    targets.append(target.detach().to(device="cpu", dtype=torch.float64))
    finally:
        model.train(was_training)
    if not logits:
        raise ValueError("temperature calibration needs at least one labeled validation sample")
    values, target = torch.cat(logits), torch.cat(targets)

    def nll(log_t):
        scaled = values / log_t.clamp(-4.0, 4.0).exp()
        return -(target * F.log_softmax(scaled, dim=-1)).sum(-1).mean()

    start = model.log_temperature.detach().to(device="cpu", dtype=torch.float64).clamp(-4.0, 4.0)
    before = nll(start).item()
    with torch.enable_grad():
        log_t = start.clone().requires_grad_()
        optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = nll(log_t)
            loss.backward()
            return loss

        optimizer.step(closure)
    with torch.no_grad():
        fitted = log_t.clamp(-4.0, 4.0).to(model.log_temperature.dtype).double()
        after = nll(fitted).item()
        if not after <= before:
            fitted, after = start, before
        model.log_temperature.copy_(fitted)
    return {"temperature": model.log_temperature.exp().item(), "nll_before": before, "nll_after": after}
