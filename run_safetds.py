from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detox_sampler import (
    DetoxESMCConfig,
    ToxicityScorer,
    detox_esmc_sample,
    set_seed,
)


MODEL_ALIASES: Dict[str, str] = {
    "gpt2-xl": "openai-community/gpt2-xl",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen-2_5-7b": "Qwen/Qwen2.5-7B"
}


def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def torch_dtype_from_arg(dtype_arg: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer_safely(args: argparse.Namespace):
    model_path = resolve_model_name(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch_dtype_from_arg(args.dtype)

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print("\n========== CUDA Check ==========")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"visible cuda device 0:     {torch.cuda.get_device_name(0)}")
    print("================================\n")

    print(f"[load] model path: {model_path}")
    print(f"[load] device:     {device}")
    print(f"[load] dtype:      {torch_dtype}")

    tok_kwargs: Dict[str, Any] = {
        "use_fast": True,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if token:
        tok_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token.")

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if token:
        model_kwargs["token"] = token
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map and args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if args.device_map == "none":
        model.to(device)
    model.eval()

    try:
        model.config.use_cache = True
    except Exception:
        pass

    return model, tokenizer, device


def make_config(args: argparse.Namespace, seed: int) -> DetoxESMCConfig:
    if args.method == "sample":
        detox_beta = 0.0
        n_particles = 1
        max_particles = 1
    elif args.method == "safe_tds":
        detox_beta = args.detox_beta
        n_particles = args.n_particles
        max_particles = args.max_particles
    else:
        raise ValueError(f"Unknown method: {args.method}")

    entropy_threshold = paper_entropy_threshold(args)

    return DetoxESMCConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens_to_keep=args.min_tokens_to_keep,
        repetition_penalty=args.repetition_penalty,
        detox_beta=detox_beta,
        prefix_epsilon=args.prefix_epsilon,
        prefix_beta_mode=args.prefix_beta_mode,
        lm_alpha=args.lm_alpha,
        safety_min_prob=args.safety_min_prob,
        score_context=args.score_context,
        n_particles=n_particles,
        block_size=args.block_size,
        entropy_resample_threshold=entropy_threshold,
        max_particles=max_particles,
        expansion_eta=args.expansion_eta,
        seed=seed,
        use_kv_cache=True,
        kv_cache_mode=args.kv_cache_mode,
        kv_page_size=args.kv_page_size,
    )


def paper_entropy_threshold(args: argparse.Namespace) -> float:
    if args.entropy_resample_threshold is not None:
        return float(args.entropy_resample_threshold)
    return 0.4 if args.dataset == "RTP-Broad" else 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="RTP-Broad", choices=["RTP-Broad", "RTP-Extreme"])
    p.add_argument("--filepath", type=str, default="save_data", help="Root output directory.")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing RTP-Broad.jsonl and RTP-Extreme.jsonl.")

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        help="Model alias, Hugging Face id, or local model directory.",
    )
    p.add_argument("--tox_model", type=str, default="unitary/toxic-bert")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="none", choices=["none", "auto", "balanced", "balanced_low_0", "sequential"])
    p.add_argument("--attn_implementation", type=str, default="auto")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")

    # Runtime / data size
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)

    # Method
    p.add_argument("--method", type=str, default="safe_tds", choices=["sample", "safe_tds"])

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--min_tokens_to_keep", type=int, default=1)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    # SafeTDS / blockwise trajectory importance sampling
    p.add_argument("--detox_beta", type=float, default=8.0, help="lambda in beta(x)=lambda*(epsilon+(1-epsilon)*tox(x)).")
    p.add_argument("--prefix_epsilon", type=float, default=0.5)
    p.add_argument(
        "--prefix_beta_mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "constant_lambda", "constant_epsilon"],
        help="Main paper uses adaptive. Ablations use constant_lambda and constant_epsilon.",
    )
    p.add_argument("--lm_alpha", type=float, default=0.9)
    p.add_argument("--safety_min_prob", type=float, default=1e-6, help="kappa in s_phi(y)=log(max(1-tox_phi(y), kappa)).")
    p.add_argument("--score_context", action="store_true")
    p.add_argument("--n_particles", type=int, default=16)
    p.add_argument("--block_size", type=int, default=5, help="Tokens per block. With max_new_tokens=20, block_size=5 gives K=4.")
    p.add_argument("--entropy_resample_threshold", type=float, default=None, help="tau for r_k^H < tau. Defaults: 0.4 for RTP-Broad, 0.5 for RTP-Extreme.")
    p.add_argument("--max_particles", type=int, default=32, help="N_max. Paper setting is 2N=32 when N=16.")
    p.add_argument("--expansion_eta", type=float, default=1.0 / 3.0)
    p.add_argument("--kv_cache_mode", type=str, default="paged", choices=["paged", "hf"], help="paged matches the paper's page-metadata prefix sharing; hf uses ordinary batch cache selection.")
    p.add_argument("--kv_page_size", type=int, default=16)

    # Toxicity scorer speed
    p.add_argument("--tox_batch_size", type=int, default=64)

    return p.parse_args()


def make_output_path(args: argparse.Namespace) -> Path:
    filepath = Path(args.filepath)

    temp_str = f"{args.temperature:.2f}".rstrip("0").rstrip(".")
    top_p_str = f"{args.top_p:.2f}".rstrip("0").rstrip(".")
    decode_dir = f"temperature_{temp_str}_top_p_{top_p_str}"

    sample_name = "all" if args.max_samples is None else str(args.max_samples)

    if args.method == "sample":
        method_prefix = "sample"
    else:
        method_prefix = (
            f"safe_tds_beta_{args.detox_beta:g}"
            f"_block_{args.block_size}"
            f"_particles_{args.n_particles}"
            f"_maxp_{args.max_particles}"
            f"_tau_{paper_entropy_threshold(args):g}"
        )

    return filepath / args.model / decode_dir / f"{method_prefix}_{args.dataset}_{sample_name}.json"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.filepath)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n========== SafeTDS Fast Settings ==========")
    print(f"output dir:             {output_root}")
    print(f"dataset:                {args.dataset}")
    print(f"model:                  {args.model}")
    print(f"method:                 {args.method}")
    print(f"k:                      {args.k}")
    print(f"max_new_tokens:         {args.max_new_tokens}")
    print(f"temperature:            {args.temperature}")
    print(f"top_p:                  {args.top_p}")
    print(f"detox_beta:             {args.detox_beta}")
    print(f"prefix epsilon:         {args.prefix_epsilon}")
    print(f"prefix beta mode:       {args.prefix_beta_mode}")
    print(f"lm_alpha:               {args.lm_alpha}")
    print(f"block_size:             {args.block_size}")
    print(f"n_particles:            {args.n_particles}")
    print(f"max_particles:          {args.max_particles}")
    print(f"expansion eta:          {args.expansion_eta}")
    print(f"entropy tau:            {paper_entropy_threshold(args)}")
    print(f"kv cache mode:          {args.kv_cache_mode}")
    print(f"kv page size:           {args.kv_page_size}")
    print(f"tox_batch_size:         {args.tox_batch_size}")
    print("==========================================\n")

    model_start = time.perf_counter()
    model, tokenizer, device = load_model_and_tokenizer_safely(args)
    print(f"[time] model loading: {time.perf_counter() - model_start:.2f}s")

    scorer_start = time.perf_counter()
    scorer = ToxicityScorer(model_name=args.tox_model, device=device, batch_size=args.tox_batch_size, use_cache=True)
    print(f"[time] scorer loading: {time.perf_counter() - scorer_start:.2f}s")

    dataset = pd.read_json(Path(args.data_dir) / f"{args.dataset}.jsonl", lines=True)
    prompts = pd.json_normalize(dataset["prompt"])["text"]
    if args.max_samples is not None:
        prompts = prompts.sample(
            n=min(args.max_samples, len(prompts)),
            random_state=args.seed,
        ).reset_index(drop=True)

    run_seed = int(args.seed)
    cfg = make_config(args, seed=run_seed)

    error_count = 0
    generations: List[List[str]] = []
    stats_all: List[List[Dict[str, Any]]] = []

    for i, gen_prompt in enumerate(tqdm(prompts, desc="safe-tds-fast-generate")):
        generation_list: List[str] = []
        stats_list: List[Dict[str, Any]] = []

        for j in range(args.k):
            cfg.seed = int(run_seed + i * args.k + j)
            try:
                result = detox_esmc_sample(
                    model=model,
                    tokenizer=tokenizer,
                    scorer=scorer,
                    prompt=gen_prompt,
                    cfg=cfg,
                    device=device,
                )
                generation = result.get("generation", "")
                stats_list.append(result.get("stats", {}))
            except RuntimeError as e:
                error_count += 1
                print(f"[warning] error at prompt={i}, sample={j}: {repr(e)}")
                generation = ""
                stats_list.append({"error": repr(e)})

            generation_list.append(generation)

        generations.append(generation_list)
        stats_all.append(stats_list)

    saved_config = vars(args).copy()
    saved_config["entropy_resample_threshold_effective"] = paper_entropy_threshold(args)
    saved_config["block_horizon_effective"] = (args.max_new_tokens + args.block_size - 1) // args.block_size

    save_data = {
        "prompts": prompts.tolist(),
        "generations": generations,
        "method": args.method,
        "model": args.model,
        "config": saved_config,
        "stats": stats_all,
    }

    out_path = make_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    print(f"[done] saved to {out_path}")
    print(f"[done] error_count={error_count}")


if __name__ == "__main__":
    main()
