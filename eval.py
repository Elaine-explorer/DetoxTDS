import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from evaluation import PerplexityEvaluator, PerspectiveToxicityEvaluator


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(file_path, datas):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)


def is_2d(values: List[Any]) -> bool:
    return bool(values) and isinstance(values[0], list)


def flatten_with_shape(values: List[Any]):
    if is_2d(values):
        shape = [len(x) for x in values]
        flat = [item for group in values for item in group]
        return flat, shape

    return values, [1] * len(values)


def reshape_by_shape(flat_values: List[Any], shape: List[int]):
    result = []
    start = 0

    for size in shape:
        end = start + size
        result.append(flat_values[start:end])
        start = end


    return result


def reshape_partial(flat_values: List[Any], shape: List[int]):
    result = []
    start = 0
    total = len(flat_values)

    for size in shape:
        if start >= total:
            break

        end = min(start + size, total)
        result.append(flat_values[start:end])
        start = end

    return result


def repeat_prompts_by_shape(prompts: List[Any], shape: List[int]):


    repeated = []
    for prompt, size in zip(prompts, shape):
        repeated.extend([prompt] * size)

    return repeated


def get_api_key():
    api_key = os.getenv("PERSPECTIVE_API_KEY")


    return api_key


def build_toxicity_evaluator(args):
    return PerspectiveToxicityEvaluator(
        api_key=get_api_key(),
        quota_per_minute=args.quota_per_minute,
        quota_safety=args.quota_safety,
        rate_limit=args.inner_batch_size,
        num_threads=args.num_threads,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )


def EvalToxicity(args):
    toxicity_evaluator = build_toxicity_evaluator(args)

    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if not file.endswith(".json") or "summary" in file or "prompts" in file:
                continue

            file_path = os.path.join(root, file)

            datas = load_data(file_path)

            if "generations" not in datas:
                continue

            generations = datas["generations"]
            flat_generations, shape = flatten_with_shape(generations)

            existing_scores = datas.get("toxicity_scores", [])

            if existing_scores:
                if is_2d(existing_scores):
                    flat_scores, score_shape = flatten_with_shape(existing_scores)
                    toxicity_scores = flat_scores
                else:
                    toxicity_scores = existing_scores

                start_index = len(toxicity_scores)

            else:
                toxicity_scores = []
                start_index = 0

            for i in range(start_index, len(flat_generations), args.outer_batch_size):
                end_i = min(i + args.outer_batch_size, len(flat_generations))

                print(f"toxicity progress: {i}/{len(flat_generations)}")

                batch_generations = flat_generations[i:end_i]
                results = toxicity_evaluator.evaluate_batch(batch_generations)
                batch_scores = [s.toxicity for s in results]

                toxicity_scores.extend(batch_scores)

                datas["toxicity_scores"] = reshape_partial(toxicity_scores, shape)
                save_data(file_path, datas)

            datas["toxicity_scores"] = reshape_by_shape(toxicity_scores, shape)
            save_data(file_path, datas)



def EvalPPL(args):
    perplexity_evaluator = PerplexityEvaluator(
        model_id=args.ppl_model_id,
        max_length=args.ppl_max_length,
        use_amp=not args.no_amp,
    )

    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if not file.endswith(".json") or "summary" in file or "prompts" in file:
                continue

            file_path = os.path.join(root, file)

            datas = load_data(file_path)

            if "generations" not in datas or "prompts" not in datas:
                continue

            prompts = datas["prompts"]
            generations = datas["generations"]

            flat_generations, shape = flatten_with_shape(generations)
            flat_prompts = repeat_prompts_by_shape(prompts, shape)

            required_keys = [
                "perplexity_scores_continuation",
                "perplexity_scores_full",
                "perplexity_scores_generation",
            ]

            has_all = True
            for key in required_keys:
                if key not in datas:
                    has_all = False
                    break

                scores = datas[key]
                flat_scores, score_shape = flatten_with_shape(scores)

                if score_shape != shape or len(flat_scores) != len(flat_generations):
                    has_all = False
                    break

            ppl_results = perplexity_evaluator.evaluate_all_modes_batch(
                prompts=flat_prompts,
                completions=flat_generations,
                batch_size=args.ppl_batch_size,
            )

            datas["perplexity_scores_continuation"] = reshape_by_shape(
                ppl_results["continuation"],
                shape,
            )
            datas["perplexity_scores_full"] = reshape_by_shape(
                ppl_results["full"],
                shape,
            )
            datas["perplexity_scores_generation"] = reshape_by_shape(
                ppl_results["generation"],
                shape,
            )
            datas["perplexity_scores"] = datas["perplexity_scores_continuation"]

            save_data(file_path, datas)


def main():
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./save_data")

    parser.add_argument("--skip_toxicity", action="store_true")
    parser.add_argument("--skip_ppl", action="store_true")

    parser.add_argument("--quota_per_minute", type=int, default=3000)
    parser.add_argument("--quota_safety", type=float, default=0.8)
    parser.add_argument("--inner_batch_size", type=int, default=10)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--outer_batch_size", type=int, default=1000)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=180)

    parser.add_argument("--ppl_batch_size", type=int, default=8)
    parser.add_argument("--ppl_max_length", type=int, default=1024)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--force_recompute_ppl", action="store_true")

    parser.add_argument(
        "--ppl_model_id",
        default="meta-llama/Llama-2-7b-hf",
    )

    args = parser.parse_args()

    if not args.skip_toxicity:
        EvalToxicity(args)

    if not args.skip_ppl:
        EvalPPL(args)

    elapsed = time.time() - start
    t = time.gmtime(elapsed)


if __name__ == "__main__":
    main()
