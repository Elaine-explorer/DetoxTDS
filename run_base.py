from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ALIASES: Dict[str, str] = {
    "gpt2-xl": "openai-community/gpt2-xl",
    "qwen-2_5-7b": "Qwen/Qwen2.5-7B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
}

def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype_from_arg(dtype_arg: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    return "auto"


def model_input_device(model, fallback: str) -> torch.device:
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


def load_base_model_and_tokenizer(args: argparse.Namespace):
    model_path = resolve_model_name(args.model)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch_dtype_from_arg(args.dtype)

    tok_kwargs: Dict[str, Any] = {
        "use_fast": True,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }

    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    print(f"[load] model={model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def base_model_generate_once(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> str:
    set_seed(seed)

    input_device = model_input_device(model, device)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    input_ids = enc.input_ids.to(input_device)
    attention_mask = enc.attention_mask.to(input_device) if hasattr(enc, "attention_mask") else None

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.top_k and args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    out = model.generate(**gen_kwargs)
    continuation_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True)


@torch.no_grad()
def base_model_generate_k_fast(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: str,
    seed: int,
) -> List[str]:
    set_seed(seed)

    input_device = model_input_device(model, device)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_tokens,
    )

    input_ids = enc.input_ids.to(input_device)
    attention_mask = enc.attention_mask.to(input_device) if hasattr(enc, "attention_mask") else None

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.top_k and args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k

    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    out = model.generate(**gen_kwargs)

    input_len = input_ids.shape[1]

    generations: List[str] = []
    for output_ids in out:
        continuation_ids = output_ids[input_len:]
        text = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
        )
        generations.append(text)

    return generations

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="RTP-Broad", choices=["RTP-Broad", "RTP-Extreme"])

    p.add_argument("--filepath", type=str, default="save_data")
    p.add_argument("--data_dir", type=str, default="data", help="Directory containing RTP-Broad.jsonl and RTP-Extreme.jsonl.")

    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Model alias, Hugging Face id, or local model directory."
    )

    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu, or empty for auto.")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")

    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--k", type=int, default=25, help="Number of generations per prompt.")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = pd.read_json(Path(args.data_dir) / f"{args.dataset}.jsonl", lines=True)

    prompts = pd.json_normalize(dataset['prompt'])['text']

    model, tokenizer, device = load_base_model_and_tokenizer(args)

    generations: List[List[str]] = []
    for i, gen_prompt in enumerate(tqdm(prompts, desc="base-model-generate")):
        prompt_generations = []

        for j in range(args.k):
            fallback_seed = int(args.seed + i * args.k + j)

            generation = base_model_generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=gen_prompt,
                args=args,
                device=device,
                seed=fallback_seed,
            )

            prompt_generations.append(generation)

        generations.append(prompt_generations)

    filepath = Path(args.filepath)

    temp_str = f"{args.temperature:.2f}".rstrip("0").rstrip(".")
    top_p_str = f"{args.top_p:.2f}".rstrip("0").rstrip(".")

    decode_dir = f"temperature_{temp_str}_top_p_{top_p_str}"

    sample_name = "all" if args.max_samples is None else str(args.max_samples)

    out_path = filepath / args.model / decode_dir / f"base_once_{args.dataset}_{sample_name}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "prompts": prompts.tolist(),
        "generations": generations,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    print(f"[done] saved to {out_path}")



if __name__ == "__main__":
    main()
