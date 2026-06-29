# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random

import torch
import torch.nn.functional as F


@dataclass
class DetoxESMCConfig:
    # generation length
    max_new_tokens: int = 20
    min_new_tokens: int = 0
    max_prompt_tokens: int = 512

    # proposal sampling q_eta
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    min_tokens_to_keep: int = 1
    repetition_penalty: float = 1.0

    # prefix-heuristics target distribution
    detox_beta: float = 8.0       # lambda in beta(x)=lambda*(epsilon+(1-epsilon)*tox(x))
    prefix_epsilon: float = 0.5
    prefix_beta_mode: str = "adaptive"  # adaptive | constant_lambda | constant_epsilon
    lm_alpha: float = 0.9         # alpha in p_theta(.)^alpha
    safety_min_prob: float = 1e-6 # kappa in s_phi(y)=log(max(1-tox_phi(y), kappa))
    score_context: bool = False   # False: score continuation; True: score prompt+continuation

    # blockwise trajectory importance sampling
    n_particles: int = 16
    block_size: int = 5           # max_new_tokens=20 and block_size=5 gives K=4

    # entropy-guided adaptive resampling / optional expansion
    entropy_resample_threshold: float = 0.5  # tau in r_k^H < tau
    max_particles: int = 32                  # N_max=2N in the paper setting
    expansion_eta: float = 1.0 / 3.0         # eta in expansion factor (0, 1]

    # decoding/stopping
    force_eos_after_done: bool = True
    seed: int = 42

    # speed: enabled by default; no CLI switch needed in most experiments
    use_kv_cache: bool = True
    kv_cache_mode: str = "paged"  # paged | hf
    kv_page_size: int = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_input_device(model, fallback: str | torch.device) -> torch.device:
    try:
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            for dev in model.hf_device_map.values():
                if isinstance(dev, str) and dev not in {"cpu", "disk"}:
                    return torch.device(dev)
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(fallback)


def top_k_top_p_filtering_(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """In-place top-k / nucleus filtering for a [batch, vocab] logits tensor."""
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits.masked_fill_(logits < kth, -float("inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumprobs > top_p
        keep = max(int(min_tokens_to_keep), 1)
        sorted_mask[:, :keep] = False
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, sorted_mask)
        logits.masked_fill_(mask, -float("inf"))

    return logits


def apply_repetition_penalty_(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """HuggingFace-style repetition penalty, applied in-place."""
    if penalty is None or penalty == 1.0 or prev_tokens.numel() == 0:
        return logits

    vocab = logits.size(-1)
    safe_prev = prev_tokens.clamp(0, vocab - 1)
    appeared = torch.zeros_like(logits, dtype=torch.bool)
    appeared.scatter_(1, safe_prev, True)

    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    logits.copy_(torch.where(appeared, penalized, logits))
    return logits


def normalized_weights_from_logw(log_w: torch.Tensor) -> torch.Tensor:
    if log_w.numel() == 1:
        return torch.ones_like(log_w)
    lw = log_w - torch.logsumexp(log_w, dim=0)
    return torch.exp(lw)


def entropy_ratio_from_logw(log_w: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    """Return H_k, r_k^H = exp(H_k) / N_k, and normalized weights."""
    n = int(log_w.numel())
    if n <= 1:
        w = torch.ones_like(log_w)
        return 0.0, 1.0, w

    w = normalized_weights_from_logw(log_w)
    h = -torch.sum(w * torch.log(w.clamp_min(1e-12))).item()
    ratio = math.exp(h) / max(n, 1)
    return float(h), float(ratio), w


def multinomial_resample(w: torch.Tensor, n_samples: Optional[int] = None, generator=None) -> torch.Tensor:
    """Multinomial resampling matching the categorical kernel in the paper."""
    n = int(w.numel())
    k = n if n_samples is None else int(n_samples)
    if n == 1:
        return torch.zeros(k, dtype=torch.long, device=w.device)
    probs = w / w.sum().clamp_min(1e-12)
    return torch.multinomial(probs, k, replacement=True, generator=generator).long()


def compute_expanded_population_size(
    n: int,
    entropy_ratio: float,
    threshold: float,
    max_particles: int,
    expansion_eta: float,
) -> int:
    n = int(n)
    max_particles = max(int(max_particles), n)
    if n <= 1 or entropy_ratio >= float(threshold):
        return n

    r = max(float(entropy_ratio), 1.0 / max(n, 1))
    eta = min(max(float(expansion_eta), 0.0), 1.0)
    expanded = int(math.ceil(n * ((1.0 / r) ** eta)))
    return min(max_particles, max(n, expanded))


def _index_select_state(
    idx: torch.Tensor,
    seqs: torch.Tensor,
    done: torch.Tensor,
    log_w: torch.Tensor,
    safe_potential: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        seqs.index_select(0, idx),
        done.index_select(0, idx),
        log_w.index_select(0, idx),
        safe_potential.index_select(0, idx),
    )


def _index_select_logits(logits: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    if logits is None:
        return None
    return logits.index_select(0, idx)


def _cache_seq_dim(tensor: torch.Tensor) -> int:
    if tensor.dim() < 3:
        raise ValueError(f"Unsupported KV tensor shape: {tuple(tensor.shape)}")
    return tensor.dim() - 2


def _cache_seq_len(tensor: torch.Tensor) -> int:
    return int(tensor.size(_cache_seq_dim(tensor)))


def _normalize_legacy_past(past_key_values: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if past_key_values is None:
        return tuple()
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "PagedKVCacheManager currently expects legacy tuple/list past_key_values. "
            "Use kv_cache_mode='hf' for transformer cache classes."
        )

    normalized: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise TypeError("Unsupported past_key_values layer format.")
        key, value = layer[0], layer[1]
        if not torch.is_tensor(key) or not torch.is_tensor(value):
            raise TypeError("PagedKVCacheManager expects tensor key/value cache entries.")
        normalized.append((key, value))
    return tuple(normalized)


@dataclass
class KVPage:
    layers: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    ref_count: int = 0


class PagedKVCacheManager:
    """
    Page-based trajectory cache used by SafeTDS.
    """

    def __init__(self, page_size: int = 16) -> None:
        self.page_size = max(1, int(page_size))
        self.pages: Dict[int, KVPage] = {}
        self.trajectories: List[List[int]] = []
        self.next_page_id = 0
        self.metadata_copies = 0
        self.page_allocations = 0

    @property
    def live_pages(self) -> int:
        return len(self.pages)

    @property
    def shared_pages(self) -> int:
        return sum(1 for page in self.pages.values() if page.ref_count > 1)

    def stats(self) -> Dict[str, int]:
        return {
            "live_pages": self.live_pages,
            "shared_pages": self.shared_pages,
            "page_allocations": self.page_allocations,
            "metadata_copies": self.metadata_copies,
        }

    def _allocate_page(self, layers: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], length: int) -> int:
        page_id = self.next_page_id
        self.next_page_id += 1
        self.pages[page_id] = KVPage(layers=layers, length=int(length), ref_count=0)
        self.page_allocations += 1
        return page_id

    def _retain(self, page_ids: Sequence[int]) -> None:
        for page_id in page_ids:
            self.pages[page_id].ref_count += 1

    def _release(self, page_ids: Sequence[int]) -> None:
        for page_id in page_ids:
            page = self.pages[page_id]
            page.ref_count -= 1
            if page.ref_count <= 0:
                del self.pages[page_id]

    def initialize_from_prompt(self, prompt_past_key_values: Any, n_particles: int) -> None:
        past = _normalize_legacy_past(prompt_past_key_values)
        if not past:
            raise ValueError("Cannot initialize paged KV cache from an empty prompt cache.")

        prompt_len = _cache_seq_len(past[0][0])
        shared_page_ids: List[int] = []
        for start in range(0, prompt_len, self.page_size):
            length = min(self.page_size, prompt_len - start)
            page_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for key, value in past:
                seq_dim = _cache_seq_dim(key)
                key_page = key.narrow(seq_dim, start, length).contiguous()
                value_page = value.narrow(_cache_seq_dim(value), start, length).contiguous()
                page_layers.append((key_page, value_page))
            shared_page_ids.append(self._allocate_page(tuple(page_layers), length))

        self.trajectories = [list(shared_page_ids) for _ in range(int(n_particles))]
        for _ in range(int(n_particles)):
            self._retain(shared_page_ids)

    def materialize(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        if not self.trajectories:
            raise ValueError("Paged KV cache has no trajectories.")

        first_pages = self.trajectories[0]
        if not first_pages:
            raise ValueError("Paged KV cache trajectory has no pages.")

        num_layers = len(self.pages[first_pages[0]].layers)
        batched_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            batch_keys: List[torch.Tensor] = []
            batch_values: List[torch.Tensor] = []
            for page_ids in self.trajectories:
                key_segments = [self.pages[page_id].layers[layer_idx][0] for page_id in page_ids]
                value_segments = [self.pages[page_id].layers[layer_idx][1] for page_id in page_ids]
                seq_dim = _cache_seq_dim(key_segments[0])
                key_seq = torch.cat(key_segments, dim=seq_dim)
                value_seq = torch.cat(value_segments, dim=_cache_seq_dim(value_segments[0]))
                batch_keys.append(key_seq)
                batch_values.append(value_seq)
            batched_layers.append((torch.cat(batch_keys, dim=0), torch.cat(batch_values, dim=0)))
        return tuple(batched_layers)

    def append_last_token_from_past(self, updated_past_key_values: Any) -> None:
        past = _normalize_legacy_past(updated_past_key_values)
        if not past:
            return

        batch_size = int(past[0][0].size(0))
        if batch_size != len(self.trajectories):
            raise ValueError("Updated KV cache batch size does not match trajectory count.")

        for row in range(batch_size):
            page_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for key, value in past:
                key_seq_dim = _cache_seq_dim(key)
                value_seq_dim = _cache_seq_dim(value)
                key_row = key.narrow(0, row, 1)
                value_row = value.narrow(0, row, 1)
                key_token = key_row.narrow(key_seq_dim, _cache_seq_len(key_row) - 1, 1).contiguous()
                value_token = value_row.narrow(value_seq_dim, _cache_seq_len(value_row) - 1, 1).contiguous()
                page_layers.append((key_token, value_token))
            page_id = self._allocate_page(tuple(page_layers), 1)
            self.trajectories[row].append(page_id)
            self._retain([page_id])

    def resample(self, idx: torch.Tensor) -> None:
        old_trajectories = self.trajectories
        new_trajectories = [list(old_trajectories[int(i)]) for i in idx.detach().cpu().tolist()]
        for page_ids in new_trajectories:
            self._retain(page_ids)
        for page_ids in old_trajectories:
            self._release(page_ids)
        self.metadata_copies += sum(len(page_ids) for page_ids in new_trajectories)
        self.trajectories = new_trajectories


def _index_select_past_key_values(past_key_values: Any, idx: torch.Tensor) -> Any:
    """Select batch dimension for HF past_key_values after resampling/expansion."""
    if past_key_values is None:
        return None

    # New cache classes in recent transformers sometimes implement batch_select_indices.
    if hasattr(past_key_values, "batch_select_indices"):
        try:
            return past_key_values.batch_select_indices(idx)
        except Exception:
            pass

    selected = []
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            selected_layer = []
            for item in layer:
                if torch.is_tensor(item) and item.size(0) >= idx.max().item() + 1:
                    selected_layer.append(item.index_select(0, idx))
                else:
                    selected_layer.append(item)
            selected.append(tuple(selected_layer))
        else:
            selected.append(layer)
    return tuple(selected)


class ToxicityScorer:
    """
    Wrapper around a HuggingFace sequence-classification toxicity model.
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.use_cache = bool(use_cache)
        self._cache: Dict[str, float] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.labels = [str(id2label.get(i, f"LABEL_{i}")).lower() for i in range(self.model.config.num_labels)]

    def _toxicity_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) == 1:
            return torch.sigmoid(logits.squeeze(-1))

        sigmoid_probs = torch.sigmoid(logits)
        toxic_keywords = (
            "toxic", "severe", "obscene", "threat", "insult",
            "identity", "hate", "offensive", "abusive", "unsafe",
        )
        safe_keywords = ("non", "not", "neutral", "normal", "clean", "safe")

        toxic_cols = []
        safe_cols = []
        for i, label in enumerate(self.labels):
            label_l = label.lower().replace("-", "_").replace(" ", "_")
            if any(k in label_l for k in safe_keywords):
                safe_cols.append(i)
            if any(k in label_l for k in toxic_keywords) and not any(k in label_l for k in safe_keywords):
                toxic_cols.append(i)

        if toxic_cols:
            return sigmoid_probs[:, toxic_cols].max(dim=-1).values
        if safe_cols:
            sm = F.softmax(logits, dim=-1)
            return 1.0 - sm[:, safe_cols].sum(dim=-1).clamp(0.0, 1.0)
        return F.softmax(logits, dim=-1).max(dim=-1).values

    @torch.inference_mode()
    def score(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        normalized = [t if t and t.strip() else " " for t in texts]
        output: List[Optional[float]] = [None] * len(normalized)

        to_compute: List[str] = []
        positions: List[int] = []
        for i, text in enumerate(normalized):
            if self.use_cache and text in self._cache:
                output[i] = self._cache[text]
            else:
                to_compute.append(text)
                positions.append(i)

        computed_values: List[float] = []
        for start in range(0, len(to_compute), self.batch_size):
            batch = to_compute[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            tox = self._toxicity_from_logits(logits).detach().float().cpu().tolist()
            computed_values.extend(float(x) for x in tox)

        for pos, text, value in zip(positions, to_compute, computed_values):
            output[pos] = value
            if self.use_cache:
                self._cache[text] = value

        return torch.tensor(output, dtype=torch.float32, device=self.device)


def _decode_continuations(tokenizer, seqs: torch.Tensor, prompt_len: int, cur: int) -> List[str]:
    pieces = seqs[:, prompt_len:cur].detach().cpu().tolist()
    return tokenizer.batch_decode(pieces, skip_special_tokens=True)


def _safe_log_from_toxicity(tox: torch.Tensor, kappa: float = 1e-6) -> torch.Tensor:
    safe_prob = (1.0 - tox.clamp(0.0, 1.0)).clamp_min(float(kappa))
    return torch.log(safe_prob)


def prefix_beta_from_toxicity(prefix_toxicity: torch.Tensor, cfg: DetoxESMCConfig) -> float:
    """Compute beta(x) from the paper, with ablation modes from Table II."""
    lam = float(cfg.detox_beta)
    eps = float(cfg.prefix_epsilon)
    tox = float(prefix_toxicity.item())
    mode = str(cfg.prefix_beta_mode)
    if mode == "adaptive":
        return lam * (eps + (1.0 - eps) * tox)
    if mode == "constant_lambda":
        return lam
    if mode == "constant_epsilon":
        return lam * eps
    raise ValueError(f"Unknown prefix_beta_mode: {cfg.prefix_beta_mode}")


def _model_forward(model, *, input_ids: torch.Tensor, use_cache: bool, past_key_values: Any = None, force_legacy_cache: bool = False):
    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": use_cache,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if force_legacy_cache:
        kwargs["return_legacy_cache"] = True
    try:
        return model(**kwargs)
    except TypeError:
        if "return_legacy_cache" in kwargs:
            kwargs.pop("return_legacy_cache")
            return model(**kwargs)
        raise


@torch.inference_mode()
def detox_esmc_sample(
    model,
    tokenizer,
    scorer: ToxicityScorer,
    prompt: str,
    cfg: DetoxESMCConfig,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(int(cfg.seed))
    input_device = model_input_device(model, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg.max_prompt_tokens))
    input_ids = enc.input_ids.to(input_device)
    prompt_len = int(input_ids.size(1))
    pad_id = int(tokenizer.pad_token_id)

    n0 = max(1, int(cfg.n_particles))
    max_particles = max(n0, int(cfg.max_particles))
    total_len = prompt_len + int(cfg.max_new_tokens)

    seqs = torch.full((n0, total_len), pad_id, dtype=torch.long, device=input_device)
    seqs[:, :prompt_len] = input_ids.expand(n0, -1)
    done = torch.zeros(n0, dtype=torch.bool, device=input_device)
    log_w = torch.zeros(n0, dtype=torch.float32, device=input_device)
    safe_potential = torch.zeros(n0, dtype=torch.float32, device=input_device)

    cur = prompt_len
    n = n0
    generated = 0
    block_step = 0

    generator = torch.Generator(device=input_device)
    generator.manual_seed(int(cfg.seed))

    prefix_toxicity = scorer.score([prompt]).to(input_device)[0]
    beta_x = prefix_beta_from_toxicity(prefix_toxicity, cfg)
    use_hf_cache = bool(cfg.use_kv_cache) and str(cfg.kv_cache_mode) == "hf"
    use_paged_cache = bool(cfg.use_kv_cache) and str(cfg.kv_cache_mode) == "paged"
    if bool(cfg.use_kv_cache) and not (use_hf_cache or use_paged_cache):
        raise ValueError(f"Unknown kv_cache_mode: {cfg.kv_cache_mode}")

    stats: Dict[str, Any] = {
        "config": asdict(cfg),
        "prefix_toxicity": float(prefix_toxicity.item()),
        "prefix_epsilon": float(cfg.prefix_epsilon),
        "prefix_beta_mode": str(cfg.prefix_beta_mode),
        "beta_x": beta_x,
        "events": [(0, n, n, "init")],
        "trajectory_entropy": [],
        "entropy_ratio": [],
        "toxicity_mean": [],
        "toxicity_min": [],
        "resample_count": 0,
        "expand_count": 0,
        "done_at": None,
        "used_kv_cache": bool(cfg.use_kv_cache),
        "kv_cache_mode": str(cfg.kv_cache_mode) if bool(cfg.use_kv_cache) else "none",
        "kv_cache_resampling": "page_metadata_refcount" if use_paged_cache else "hf_batch_select_indices_or_index_select",
        "kv_cache_page_stats": [],
    }

    # Initial forward over the prompt. Then decode one token at a time from cache.
    past_key_values = None
    paged_cache: Optional[PagedKVCacheManager] = None
    if use_paged_cache:
        out = _model_forward(
            model,
            input_ids=input_ids,
            use_cache=True,
            force_legacy_cache=True,
        )
        logits = out.logits[:, -1, :].float().expand(n0, -1).contiguous()
        paged_cache = PagedKVCacheManager(page_size=int(cfg.kv_page_size))
        paged_cache.initialize_from_prompt(out.past_key_values, n_particles=n0)
        stats["kv_cache_page_stats"].append((0, generated, paged_cache.stats()))
    elif use_hf_cache:
        out = _model_forward(model, input_ids=seqs[:, :prompt_len], use_cache=True)
        logits = out.logits[:, -1, :].float()
        past_key_values = out.past_key_values
    else:
        logits = None

    while generated < int(cfg.max_new_tokens):
        block_len = min(max(int(cfg.block_size), 1), int(cfg.max_new_tokens) - generated)
        block_logp = torch.zeros(n, dtype=torch.float32, device=input_device)
        block_logq = torch.zeros(n, dtype=torch.float32, device=input_device)

        for _ in range(block_len):
            if not bool(cfg.use_kv_cache):
                out = model(input_ids=seqs[:, :cur], use_cache=False)
                logits = out.logits[:, -1, :].float()

            assert logits is not None
            base_logprobs = F.log_softmax(logits, dim=-1)

            temp = max(float(cfg.temperature), 1e-6)
            prop_logits = logits / temp

            if cfg.repetition_penalty and float(cfg.repetition_penalty) != 1.0:
                history = seqs[:, prompt_len:cur]
                apply_repetition_penalty_(prop_logits, history, float(cfg.repetition_penalty))

            if generated < int(cfg.min_new_tokens):
                prop_logits[:, eos_id] = -float("inf")

            top_k_top_p_filtering_(
                prop_logits,
                top_k=int(cfg.top_k),
                top_p=float(cfg.top_p),
                min_tokens_to_keep=int(cfg.min_tokens_to_keep),
            )
            prop_logprobs = F.log_softmax(prop_logits, dim=-1)
            prop_probs = torch.softmax(prop_logits, dim=-1)

            next_tokens = torch.empty(n, dtype=torch.long, device=input_device)
            active = ~done
            if active.any():
                next_tokens[active] = torch.multinomial(prop_probs[active], 1, generator=generator).squeeze(-1)
            if cfg.force_eos_after_done and done.any():
                next_tokens[done] = eos_id

            gathered = next_tokens.view(n, 1)
            token_logp = torch.gather(base_logprobs, -1, gathered).squeeze(-1)
            token_logq = torch.gather(prop_logprobs, -1, gathered).squeeze(-1)

            if cfg.force_eos_after_done and done.any():
                token_logp = torch.where(done, torch.zeros_like(token_logp), token_logp)
                token_logq = torch.where(done, torch.zeros_like(token_logq), token_logq)

            block_logp = block_logp + token_logp
            block_logq = block_logq + token_logq

            seqs[:, cur] = next_tokens
            cur += 1
            generated += 1
            done = done | (next_tokens == eos_id)

            # Advance KV cache immediately so logits is ready for the next token.
            if use_paged_cache and generated < int(cfg.max_new_tokens) and not bool(done.all()):
                assert paged_cache is not None
                materialized_past = paged_cache.materialize()
                out = _model_forward(
                    model,
                    input_ids=next_tokens.view(n, 1),
                    past_key_values=materialized_past,
                    use_cache=True,
                    force_legacy_cache=True,
                )
                logits = out.logits[:, -1, :].float()
                paged_cache.append_last_token_from_past(out.past_key_values)
            elif use_hf_cache and generated < int(cfg.max_new_tokens) and not bool(done.all()):
                out = _model_forward(
                    model,
                    input_ids=next_tokens.view(n, 1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :].float()
                past_key_values = out.past_key_values

            if stats["done_at"] is None and bool(done.all()):
                stats["done_at"] = generated
                break

        block_step += 1

        # Block-level importance correction:
        # log w_k = log w_{k-1} + alpha log p_theta(block) - log q_eta(block) + beta_x Delta s_k
        log_w = log_w + float(cfg.lm_alpha) * block_logp - block_logq

        if beta_x != 0.0:
            conts = _decode_continuations(tokenizer, seqs, prompt_len, cur)
            score_texts = [(prompt + c) if cfg.score_context else c for c in conts]
            tox = scorer.score(score_texts).to(input_device)
            new_safe = _safe_log_from_toxicity(tox, cfg.safety_min_prob)
            delta_safe = new_safe - safe_potential
            log_w = log_w + beta_x * delta_safe
            safe_potential = new_safe
            stats["toxicity_mean"].append((block_step, generated, float(tox.mean().item())))
            stats["toxicity_min"].append((block_step, generated, float(tox.min().item())))

        h, ratio, w_norm = entropy_ratio_from_logw(log_w)
        stats["trajectory_entropy"].append((block_step, generated, h, n))
        stats["entropy_ratio"].append((block_step, generated, ratio, n))

        if n > 1 and ratio < float(cfg.entropy_resample_threshold):
            new_n = compute_expanded_population_size(
                n=n,
                entropy_ratio=ratio,
                threshold=float(cfg.entropy_resample_threshold),
                max_particles=max_particles,
                expansion_eta=float(cfg.expansion_eta),
            )
            idx = multinomial_resample(w_norm, n_samples=new_n, generator=generator)
            old_n = n
            seqs, done, log_w, safe_potential = _index_select_state(idx, seqs, done, log_w, safe_potential)
            if use_paged_cache:
                assert paged_cache is not None
                paged_cache.resample(idx)
                stats["kv_cache_page_stats"].append((block_step, generated, paged_cache.stats()))
            elif use_hf_cache:
                past_key_values = _index_select_past_key_values(past_key_values, idx)
            logits = _index_select_logits(logits, idx)
            n = new_n

            # After resampling, the empirical population is equally weighted.
            log_w.zero_()
            if new_n > old_n:
                stats["expand_count"] += 1
                stats["events"].append((block_step, old_n, new_n, "entropy_expand_resample"))
            else:
                stats["resample_count"] += 1
                stats["events"].append((block_step, old_n, new_n, "entropy_resample"))

        if bool(done.all()) or generated >= int(cfg.max_new_tokens):
            break

    conts = _decode_continuations(tokenizer, seqs, prompt_len, cur)
    _, _, final_w = entropy_ratio_from_logw(log_w)
    chosen_idx = int(torch.multinomial(final_w, 1, generator=generator).item()) if final_w.numel() > 1 else 0
    return {
        "generation": conts[chosen_idx],
        "chosen_idx": chosen_idx,
        "num_candidates": len(conts),
        "candidates": conts,
        "stats": stats,
    }
