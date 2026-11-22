"""Late-interaction battle critic, cached ensemble inference and disjoint team search."""
from __future__ import annotations

import inspect
import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from labyrinth_nn.catalog.schema import (
    ENEMY_STATIC_FEAT_NAMES, GLOBAL_FEAT_NAMES, TEAM_POS_CHANNELS,
    TEAM_SYNERGY_FEAT_NAMES, UNIT_STATIC_FEAT_NAMES, FeatureCatalog,
)
from labyrinth_nn.data.dataset import build_vocab_maps, collate_battle_batch, featurize_sample, team_features
from labyrinth_nn.data.sample_schema import BattleSample, NUM_OUTCOME_CLASSES, OUTCOME_WIN


@dataclass
class CriticConfig:
    """Vocabulary sizes include the shared padding/unknown index zero."""
    num_units: int
    num_enemies: int
    num_quests: int
    num_relics: int
    unit_feat_dim: int = len(UNIT_STATIC_FEAT_NAMES)
    enemy_feat_dim: int = len(ENEMY_STATIC_FEAT_NAMES)
    global_feat_dim: int = len(GLOBAL_FEAT_NAMES)
    team_synergy_dim: int = len(TEAM_SYNERGY_FEAT_NAMES)
    team_pos_channels: int = TEAM_POS_CHANNELS
    ablation_mode: str = "m2b"
    d_model: int = 128
    n_heads: int = 4
    n_context_layers: int = 2
    n_team_layers: int = 2
    dropout: float = 0.1
    relic_max_len: int = 32
    roster_max_len: int = 64
    max_areas: int = 16
    max_difficulties: int = 10
    use_global_feats: bool = True
    use_pos_bias: bool = True
    use_static_feats: bool = True
    unit_id_dropout: float = 0.15
    relic_id_mask_prob: float = 0.25
    relic_stack_noise_prob: float = 0.15
    gold_mask_prob: float = 0.55
    batch_size: int = 256
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 0.01
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    damage_loss_weight: float = 1.0
    time_loss_weight: float = 0.25
    affinity_loss_weight: float = 0.1

    def __post_init__(self):
        if self.n_heads <= 0 or self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if min(self.num_units, self.num_enemies, self.num_quests, self.num_relics, self.roster_max_len,
               self.max_areas, self.max_difficulties, self.n_context_layers, self.n_team_layers) < 1:
            raise ValueError("vocabularies, roster capacity, embedding tables and layer counts must be positive")
        if self.relic_max_len < 0:
            raise ValueError("relic_max_len must be nonnegative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], catalog: FeatureCatalog | None = None) -> CriticConfig:
        values = dict(raw)
        if catalog is not None:
            for name in ("units", "enemies", "quests", "relics"):
                key = f"num_{name}"
                size = len(getattr(catalog, name)) + 1
                if values.setdefault(key, size) != size:
                    raise ValueError(f"{key} does not match the catalog vocabulary")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path, catalog: FeatureCatalog | None = None) -> CriticConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), catalog)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    weight = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weight).sum(1) / weight.sum(1).clamp_min(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        d = config.d_model
        self.heads = config.n_heads
        self.width = d // self.heads
        self.scale = self.width ** -0.5
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.output = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.dropout)

    def _heads(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.width).transpose(1, 2)

    def forward(self, query: Tensor, keys: Tensor, allow: Tensor, bias: Tensor | None = None) -> Tensor:
        q = self._heads(self.query(query)) * self.scale
        k, v = self._heads(self.key(keys)), self._heads(self.value(keys))
        scores = torch.matmul(q, k.transpose(-2, -1))
        if bias is not None:
            scores = scores + bias
        scores = scores + (~allow.unsqueeze(1)).to(scores.dtype) * -1.0e4
        attended = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), v)
        return self.output(attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class _AttentionBlock(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        d = config.d_model
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(4 * d, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query, keys, allow, bias=None):
        x = self.norm1(query + self.dropout(self.attention(query, keys, allow, bias)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class _ContextLayer(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.self_context = _AttentionBlock(config)
        self.roster_reads_context = _AttentionBlock(config)
        self.context_reads_roster = _AttentionBlock(config)

    def forward(self, ctx, roster, ctx_mask, roster_mask, sink):
        ctx = self.self_context(ctx, ctx, ctx_mask.unsqueeze(1))
        roster = self.roster_reads_context(roster, ctx, ctx_mask.unsqueeze(1))
        roster = roster * roster_mask.unsqueeze(-1).to(roster.dtype)
        keys = torch.cat((sink, roster), dim=1)
        mask = torch.cat((torch.ones_like(ctx_mask[:, :1]), roster_mask), dim=1)
        ctx = self.context_reads_roster(ctx, keys, mask.unsqueeze(1))
        return ctx * ctx_mask.unsqueeze(-1).to(ctx.dtype), roster


class ContextRosterEncoder(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.quest_emb = nn.Embedding(config.num_quests, d, padding_idx=0)
        self.area_emb = nn.Embedding(config.max_areas, d)
        self.diff_emb = nn.Embedding(config.max_difficulties, d)
        self.enemy_emb = nn.Embedding(config.num_enemies, d, padding_idx=0)
        self.relic_emb = nn.Embedding(config.num_relics, d, padding_idx=0)
        self.unit_emb = nn.Embedding(config.num_units, d, padding_idx=0)
        if config.use_global_feats:
            self.global_proj = nn.Linear(config.global_feat_dim, d)
        if config.use_static_feats:
            self.enemy_feat_proj = nn.Linear(config.enemy_feat_dim, d)
            self.unit_feat_proj = nn.Linear(config.unit_feat_dim, d)
        self.null_relic_token = nn.Parameter(torch.empty(1, 1, d))
        self.null_roster_sink = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.null_relic_token, std=0.02)
        nn.init.normal_(self.null_roster_sink, std=0.02)
        self.relic_pool = MultiHeadAttention(config)
        self.layers = nn.ModuleList(_ContextLayer(config) for _ in range(config.n_context_layers))
        self.affinity = nn.Linear(d, 1)
        self.summary = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                roster_unit_ids, roster_mask, relic_ids, relic_mask, *,
                enemy_feats=None, roster_unit_feats=None, global_feats=None):
        c = self.config
        quest = (self.quest_emb(quest_idx) + self.area_emb(area_idx.clamp(0, c.max_areas - 1))
                 + self.diff_emb(difficulty_idx.clamp(0, c.max_difficulties - 1)))
        if c.use_global_feats:
            quest = quest + self.global_proj(global_feats.to(quest.dtype))
        enemies, roster = self.enemy_emb(enemy_ids), self.unit_emb(roster_unit_ids)
        if c.use_static_feats:
            enemies = enemies + self.enemy_feat_proj(enemy_feats.to(enemies.dtype))
            roster = roster + self.unit_feat_proj(roster_unit_feats.to(roster.dtype))
        null_relic = self.null_relic_token.expand(quest.shape[0], -1, -1)
        always = torch.ones_like(quest_idx.unsqueeze(1), dtype=torch.bool)
        relic_keys = torch.cat((null_relic, self.relic_emb(relic_ids)), dim=1)
        relic_allow = torch.cat((always, relic_mask), dim=1).unsqueeze(1)
        relic = self.relic_pool(null_relic, relic_keys, relic_allow)
        ctx = torch.cat((quest.unsqueeze(1), relic, enemies), dim=1)
        ctx_mask = torch.cat((always, always, enemy_mask), dim=1)
        sink = self.null_roster_sink.expand(quest.shape[0], -1, -1)
        for layer in self.layers:
            ctx, roster = layer(ctx, roster, ctx_mask, roster_mask, sink)
        affinity = self.affinity(roster).squeeze(-1) * roster_mask.to(roster.dtype)
        summary = self.summary(torch.cat((ctx[:, 0], ctx[:, 1], _masked_mean(ctx[:, 2:], enemy_mask)), dim=-1))
        return roster, affinity, summary


class SlotAwareTeamHead(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.slot_emb = nn.Embedding(5, d)
        self.register_buffer("slots", torch.arange(5), persistent=False)
        self.register_buffer("diagonal", torch.eye(5, dtype=torch.bool), persistent=False)
        if config.use_pos_bias:
            hidden = max(1, d // 2)
            self.pos_bias = nn.Sequential(nn.Linear(config.team_pos_channels, hidden), nn.GELU(),
                                          nn.Linear(hidden, config.n_heads))
        if config.use_global_feats:
            self.synergy_proj = nn.Linear(config.team_synergy_dim, d)
        self.layers = nn.ModuleList(_AttentionBlock(config) for _ in range(config.n_team_layers))
        self.fusion = nn.Sequential(nn.Linear((3 + int(config.use_global_feats)) * d, d), nn.GELU(),
                                    nn.LayerNorm(d), nn.Dropout(config.dropout))
        self.outcome = nn.Linear(d, NUM_OUTCOME_CLASSES)
        self.damage_mean = nn.Linear(d, 1)
        self.damage_log_std = nn.Linear(d, 1)
        self.time = nn.Linear(d, 1)

    def forward(self, roster_ctx, context_summary, team_roster_pos, team_slot_mask, *,
                team_pos_matrix=None, team_synergy=None):
        indices = team_roster_pos.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        team = torch.gather(roster_ctx, 1, indices) + self.slot_emb(self.slots).unsqueeze(0)
        weight = team_slot_mask.unsqueeze(-1).to(team.dtype)
        team = team * weight
        allow = team_slot_mask.unsqueeze(1).expand(-1, 5, -1) | self.diagonal
        bias = None
        if self.config.use_pos_bias:
            # An unknown unit's sort sentinel produces a signed gap of almost 1e6.
            # Bound it before autocast's float16 Linear, preserving all in-game distances.
            positions = team_pos_matrix.clamp(-1000.0, 1000.0).to(self.slot_emb.weight.dtype)
            bias = self.pos_bias(positions).permute(0, 3, 1, 2)
        for layer in self.layers:
            team = layer(team, team, allow, bias) * weight
        parts = [_masked_mean(team, team_slot_mask), team[:, 0], context_summary]
        if self.config.use_global_feats:
            # The same sentinel can appear in the front gap even without attention bias (m1).
            gap = TEAM_SYNERGY_FEAT_NAMES.index("front_gap_norm")
            synergy = torch.cat((team_synergy[:, :gap], team_synergy[:, gap:gap + 1].clamp(-1000.0, 1000.0),
                                 team_synergy[:, gap + 1:]), dim=1)
            parts.append(self.synergy_proj(synergy.to(self.synergy_proj.weight.dtype)))
        fused = self.fusion(torch.cat(parts, dim=-1))
        return (self.outcome(fused), self.damage_mean(fused), self.damage_log_std(fused).clamp(-5.0, 1.0),
                torch.sigmoid(self.time(fused)))


_CONTEXT_IDS = ("quest_idx", "area_idx", "difficulty_idx", "enemy_ids", "enemy_mask",
                "roster_unit_ids", "roster_mask", "relic_ids", "relic_mask")


class TwoTowerBattleCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        self.tower_a = ContextRosterEncoder(config)
        self.tower_b = SlotAwareTeamHead(config)
        self.register_buffer("log_temperature", torch.tensor(0.0, dtype=torch.float32))

    def calibrated_probs(self, logits: Tensor) -> Tensor:
        return F.softmax(logits / self.log_temperature.clamp(-4.0, 4.0).exp(), dim=-1)

    def encode_context(self, batch):
        extra = {}
        if self.config.use_static_feats:
            extra.update(enemy_feats=batch["enemy_feats"], roster_unit_feats=batch["roster_unit_feats"])
        if self.config.use_global_feats:
            extra["global_feats"] = batch["global_feats"]
        return self.tower_a(*(batch[key] for key in _CONTEXT_IDS), **extra)

    def score_team(self, roster_ctx, context_summary, batch):
        extra = {}
        if self.config.use_pos_bias:
            extra["team_pos_matrix"] = batch["team_pos_matrix"]
        if self.config.use_global_feats:
            extra["team_synergy"] = batch["team_synergy"]
        return self.tower_b(roster_ctx, context_summary, batch["team_roster_pos"], batch["team_slot_mask"], **extra)

    def forward(self, batch: Mapping[str, Any] | None = None, **inputs) -> dict[str, Tensor]:
        features = inputs if batch is None else batch
        roster, affinity, context = self.encode_context(features)
        logits, mean_logit, log_std, time = self.score_team(roster, context, features)
        return {"outcome_logits": logits, "outcome_probs": self.calibrated_probs(logits),
                "damage_mean_logit": mean_logit, "damage_mean": mean_logit.sigmoid(),
                "damage_log_std": log_std, "damage_std": log_std.exp(), "win_rem_time": time,
                "unit_affinity": affinity, "roster_ctx": roster, "context_summary": context}


class TowerAOnnxWrapper(nn.Module):
    output_names = ("roster_ctx", "unit_affinity", "context_summary")

    def __init__(self, critic: TwoTowerBattleCritic):
        super().__init__()
        self.encoder = critic.tower_a
        variants = {(False, False): self._ids, (True, False): self._static,
                    (False, True): self._global, (True, True): self._all_features}
        self.forward = variants[critic.config.use_static_feats, critic.config.use_global_feats]
        self.input_names = tuple(inspect.signature(self.forward).parameters)

    def _ids(self, quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
             roster_unit_ids, roster_mask, relic_ids, relic_mask):
        return self.encoder(quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                            roster_unit_ids, roster_mask, relic_ids, relic_mask)

    def _static(self, quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                roster_unit_ids, roster_mask, relic_ids, relic_mask, enemy_feats, roster_unit_feats):
        return self.encoder(quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                            roster_unit_ids, roster_mask, relic_ids, relic_mask,
                            enemy_feats=enemy_feats, roster_unit_feats=roster_unit_feats)

    def _global(self, quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                roster_unit_ids, roster_mask, relic_ids, relic_mask, global_feats):
        return self.encoder(quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                            roster_unit_ids, roster_mask, relic_ids, relic_mask, global_feats=global_feats)

    def _all_features(self, quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                      roster_unit_ids, roster_mask, relic_ids, relic_mask, enemy_feats, roster_unit_feats, global_feats):
        return self.encoder(quest_idx, area_idx, difficulty_idx, enemy_ids, enemy_mask,
                            roster_unit_ids, roster_mask, relic_ids, relic_mask,
                            enemy_feats=enemy_feats, roster_unit_feats=roster_unit_feats, global_feats=global_feats)


class TowerBOnnxWrapper(nn.Module):
    output_names = ("outcome_probs", "damage_mean", "damage_std", "win_rem_time")

    def __init__(self, critic: TwoTowerBattleCritic):
        super().__init__()
        self.critic = critic
        variants = {(False, False): self._ids, (True, False): self._position,
                    (False, True): self._synergy, (True, True): self._all_features}
        self.forward = variants[critic.config.use_pos_bias, critic.config.use_global_feats]
        self.input_names = tuple(inspect.signature(self.forward).parameters)

    def _predict(self, *args, **kwargs):
        logits, mean_logit, log_std, time = self.critic.tower_b(*args, **kwargs)
        return self.critic.calibrated_probs(logits), mean_logit.sigmoid(), log_std.exp(), time

    def _ids(self, roster_ctx, context_summary, team_roster_pos, team_slot_mask):
        return self._predict(roster_ctx, context_summary, team_roster_pos, team_slot_mask)

    def _position(self, roster_ctx, context_summary, team_roster_pos, team_slot_mask, team_pos_matrix):
        return self._predict(roster_ctx, context_summary, team_roster_pos, team_slot_mask, team_pos_matrix=team_pos_matrix)

    def _synergy(self, roster_ctx, context_summary, team_roster_pos, team_slot_mask, team_synergy):
        return self._predict(roster_ctx, context_summary, team_roster_pos, team_slot_mask, team_synergy=team_synergy)

    def _all_features(self, roster_ctx, context_summary, team_roster_pos, team_slot_mask, team_pos_matrix, team_synergy):
        return self._predict(roster_ctx, context_summary, team_roster_pos, team_slot_mask,
                             team_pos_matrix=team_pos_matrix, team_synergy=team_synergy)


class TwoTowerEnsemble(nn.Module):
    """Each member keeps its own latent space. Only affinities and decoded predictions are averaged."""
    def __init__(self, models: Sequence[TwoTowerBattleCritic], lcb_z: float = 1.0):
        super().__init__()
        if not models or not math.isfinite(lcb_z) or lcb_z < 0:
            raise ValueError("an ensemble requires at least one model and a finite nonnegative lcb_z")
        self.models = nn.ModuleList(models)
        self.lcb_z = lcb_z
        self.config = models[0].config
        fields = ("num_units", "num_enemies", "num_quests", "num_relics", "roster_max_len", "relic_max_len")
        for model in models:
            if any(getattr(model.config, k) != getattr(self.config, k) for k in fields):
                raise ValueError("ensemble members must use matching vocabularies and input capacities")
        self.eval()

    @torch.no_grad()
    def encode_context(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        members, affinities = [], []
        for model in self.models:
            roster, affinity, summary = model.encode_context(batch)
            members.append((roster, summary))
            affinities.append(affinity)
        return {"members": members, "unit_affinity": torch.stack(affinities).mean(0)}

    @torch.no_grad()
    def score_teams(self, cache, batch, context_indices: Tensor | None = None) -> dict[str, Tensor]:
        """Broadcast a single cached context, or select a context per team by index."""
        size = batch["team_roster_pos"].shape[0]
        probs, damages, variances, times = [], [], [], []
        for model, (roster, summary) in zip(self.models, cache["members"]):
            if context_indices is not None:
                roster = roster.index_select(0, context_indices)
                summary = summary.index_select(0, context_indices)
            elif roster.shape[0] == 1:
                roster, summary = roster.expand(size, -1, -1), summary.expand(size, -1)
            elif roster.shape[0] != size:
                raise ValueError("supply context_indices when candidate and context batches differ")
            logits, mean_logit, log_std, time = model.score_team(roster, summary, batch)
            probs.append(model.calibrated_probs(logits))
            damages.append(mean_logit.sigmoid())
            variances.append((2 * log_std).exp())
            times.append(time)
        probs, damages, times = torch.stack(probs), torch.stack(damages), torch.stack(times)
        p, p_std = probs.mean(0), probs.std(0, unbiased=False)
        win, win_std = p[:, OUTCOME_WIN], p_std[:, OUTCOME_WIN]
        damage_var = damages.var(0, unbiased=False)
        return {
            "outcome_probs": p, "outcome_probs_std": p_std, "p_win_mean": win, "p_win_std": win_std,
            "lcb_win": (win - self.lcb_z * win_std).clamp(0, 1),
            "damage_mean": damages.mean(0), "damage_mean_std": damage_var.sqrt(),
            "damage_std": (torch.stack(variances).mean(0) + damage_var).sqrt(),
            "win_rem_time": times.mean(0), "win_rem_time_std": times.std(0, unbiased=False),
        }

    def forward(self, batch):
        cache = self.encode_context(batch)
        return {**self.score_teams(cache, batch), "unit_affinity": cache["unit_affinity"]}


def _team_sig(ids: Iterable[int]) -> str:
    return ",".join(map(str, sorted(ids)))


class SingleTeamPriorityRanker:
    def __init__(self, ensemble: TwoTowerEnsemble, catalog: FeatureCatalog):
        self.ensemble = ensemble
        self.catalog = catalog
        self.maps = build_vocab_maps(catalog)

    @staticmethod
    def priority(team: Mapping[str, Any]) -> tuple[float, float, float]:
        win = team["lcb_win"]
        return win, team["win_rem_time"], 1e-6 * team["damage_mean"] if win <= 1e-3 else 0.0

    @staticmethod
    def pack_teams(teams: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Exact search on the evaluated pool for up to three disjoint teams.

        Maximize 1 - product(1 - LCB) under independent attempts, then total damage.
        Optimistic bounds ignore overlap and can only overestimate a continuation.
        """
        ordered = sorted(teams, key=lambda t: (-t["lcb_win"], -t["damage_mean"], t["team_sig"]))
        unit_bits = {uid: 1 << i for i, uid in enumerate(sorted({u for t in teams for u in t["team_unit_ids"]}))}
        masks = [sum(unit_bits[u] for u in set(t["team_unit_ids"])) for t in ordered]
        tail_damage = [0.0] * (len(ordered) + 1)
        for i in reversed(range(len(ordered))):
            tail_damage[i] = max(tail_damage[i + 1], ordered[i]["damage_mean"])
        best_score = (-1.0, 0.0, 0)
        best = []

        def search(start, used, miss, damage, chosen):
            nonlocal best_score, best
            score = (-miss, damage, len(chosen))
            if score > best_score:
                best_score, best = score, chosen
            slots = 3 - len(chosen)
            if not slots:
                return
            for i in range(start, len(ordered)):
                t = ordered[i]
                remaining = min(slots, len(ordered) - i)
                optimistic_miss, optimistic_damage = miss, damage
                for _ in range(remaining):
                    optimistic_miss *= 1 - t["lcb_win"]
                    optimistic_damage += tail_damage[i]
                if (-optimistic_miss, optimistic_damage, len(chosen) + remaining) <= best_score:
                    break
                if not used & masks[i]:
                    search(i + 1, used | masks[i], miss * (1 - t["lcb_win"]),
                           damage + t["damage_mean"], chosen + [t])

        search(0, 0, 1.0, 0.0, [])
        return best

    def _seeds(self, affinity_order: list[int]) -> Iterable[tuple[int, ...]]:
        size = min(5, len(affinity_order))
        yield tuple(affinity_order[:size])
        for offset in range(size, len(affinity_order) - size + 1, size):
            yield tuple(affinity_order[offset:offset + size])
        units = self.catalog.units
        for talent in range(1, 6):
            same = [u for u in affinity_order if u in units and units[u].talent_id == talent]
            if same:
                yield tuple((same + [u for u in affinity_order if u not in same])[:size])
        by_width = sorted(affinity_order, key=lambda u: (units[u].search_area_width if u in units else 10**9, u))
        front = set(by_width[:max(1, len(by_width) // 3)])
        vanguard = [u for u in affinity_order if u in front]
        rear = [u for u in affinity_order if u not in front]
        for count in (1, 2):
            yield tuple((vanguard[:count] + rear + vanguard[count:])[:size])

    @staticmethod
    def _swaps(seed, order, count):
        members = set(seed)
        available = [uid for uid in order if uid not in members]
        for incoming in combinations(available, count):
            for outgoing in combinations(seed, count):
                yield tuple(sorted(members.difference(outgoing).union(incoming)))

    @torch.no_grad()
    def recommend(self, sample: BattleSample, *, failed_team_signatures: Iterable[str | Sequence[int]] = (),
                  top_k: int = 10, max_candidates: int = 512, beam_size: int = 6, search_rounds: int = 2,
                  batch_size: int = 64) -> dict[str, Any]:
        if min(max_candidates, beam_size, batch_size) <= 0 or min(search_rounds, top_k) < 0:
            raise ValueError("candidate, beam and batch capacities must be positive")
        c = self.ensemble.config
        fixture = featurize_sample(self.catalog, sample, **self.maps, roster_max_len=c.roster_max_len,
                                   relic_max_len=c.relic_max_len)
        device = next(self.ensemble.parameters()).device

        def batch_on_device(rows):
            return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in collate_battle_batch(rows).items()}

        self.ensemble.eval()
        cache = self.ensemble.encode_context(batch_on_device([fixture]))
        index = {uid: i for i, uid in enumerate(fixture["roster_raw_ids"]) if uid}
        affinity = cache["unit_affinity"][0].tolist()
        order = sorted(index, key=lambda uid: (-affinity[index[uid]], uid))
        failed = {sig if isinstance(sig, str) else _team_sig(sig) for sig in failed_team_signatures}
        pool: dict[str, dict[str, Any]] = {}
        gold = fixture["global_feats"][GLOBAL_FEAT_NAMES.index("alpha_rupee_norm")]
        owned = fixture["global_feats"][GLOBAL_FEAT_NAMES.index("owned_unit_count_25_norm")]

        def evaluate(candidates, budget):
            pending = {}
            for ids in candidates:
                sig = _team_sig(ids)
                if sig not in failed and sig not in pool:
                    pending[sig] = ids
                if len(pending) >= budget:
                    break
            items = list(pending.items())
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                rows = []
                for _, ids in chunk:
                    row = team_features(self.catalog, ids, sample.relic_ids, gold, owned)
                    row["team_roster_pos"] = [index.get(uid, 0) for uid in row["team_unit_ids_sorted"]]
                    rows.append(row)
                preds = self.ensemble.score_teams(cache, batch_on_device(rows))
                names = ("lcb_win", "p_win_mean", "p_win_std", "damage_mean", "win_rem_time")
                columns = {k: preds[k].reshape(-1).tolist() for k in names}
                for j, (sig, _) in enumerate(chunk):
                    pool[sig] = {k: columns[k][j] for k in names}
                    pool[sig].update(team_sig=sig, team_unit_ids=[u for u in rows[j]["team_unit_ids_sorted"] if u])

        if order:
            evaluate(self._seeds(order), max_candidates)
            if not pool:
                evaluate(combinations(order, min(5, len(order))), 1)
        for step in range(search_rounds):
            remaining = max_candidates - len(pool)
            if not pool or remaining <= 0:
                break
            beam = sorted(pool.values(), key=self.priority, reverse=True)[:beam_size]
            streams = deque(self._swaps(team["team_unit_ids"], order, n) for team in beam for n in (1, 2))

            def neighbors():
                while streams:
                    stream = streams.popleft()
                    candidate = next(stream, None)
                    if candidate is not None:
                        yield candidate
                        streams.append(stream)

            before = len(pool)
            evaluate(neighbors(), max(1, remaining // (search_rounds - step)))
            if len(pool) == before:
                break
        ranked = sorted(pool.values(), key=self.priority, reverse=True)
        packing = self.pack_teams(ranked)
        return {"ranked_teams": ranked[:top_k], "packing": packing, "evaluated_count": len(pool),
                "joint_win_probability": 1 - math.prod(1 - t["lcb_win"] for t in packing),
                "total_damage": sum(t["damage_mean"] for t in packing)}
