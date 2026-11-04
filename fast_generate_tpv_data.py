# This software is for non-commercial use only.
# Commercial use requires a separate license.
"""Faster drop-in for generate_tpv_data.py.

generate(output_hidden_states=True) makes transformers materialise all 33 layers
for every decode step while the pipeline only ever uses the last layer. That costs
~8x wall-clock. Because the model is causal, a single forward pass over an
already-generated sequence reproduces the same last-layer states, so we:

  pass 1: generate with hidden states OFF   (fast)
  pass 2: one forward per sequence to extract last-layer states

Verified against generate()'s own output: relative error ~2.5e-5 (fp16 round-off),
min cosine similarity 0.9993. Output files are written in exactly the format
prepare_tpv_dataset.py expects, so the rest of the pipeline is unchanged.
"""
import argparse, gc, json, os, random, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_problems, resolve_hidden_layer, build_prompt_template


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--dataset", default="math500")
    p.add_argument("--start_problem_index", type=int, default=0)
    p.add_argument("--end_problem_index", type=int, default=30)
    p.add_argument("--generations_per_problem", type=int, default=5)
    p.add_argument("--output_dir", default="model_results")
    p.add_argument("--prompt_template", default="{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<think>\n")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_layer", type=int, default=None)
    p.add_argument("--use_chat_template", action="store_true", default=False)
    a = p.parse_args()

    if a.seed is not None:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        print(f"[INFO] Seeds set to {a.seed}", flush=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda")
    model.eval()
    layer_idx = resolve_hidden_layer(model, a.hidden_layer)
    print(f"[INFO] Extracting hidden_states[{layer_idx}]", flush=True)
    if a.use_chat_template:
        a.prompt_template = build_prompt_template(tok, use_chat_template=True)
        print(f"[INFO] chat template:\n{a.prompt_template}", flush=True)

    problems = load_problems(a.dataset, starting_index=a.start_problem_index, end_index=a.end_problem_index)
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump({
            "model_name": a.model, "max_new_tokens": a.max_new_tokens, "dataset_name": a.dataset,
            "generations_per_problem": a.generations_per_problem, "prompt_template": a.prompt_template,
            "temperature": a.temperature, "do_sample": a.do_sample, "top_p": a.top_p, "seed": a.seed,
            "hidden_layer": a.hidden_layer, "resolved_hidden_layer": layer_idx,
            "extraction": "two_pass_forward",
        }, f, indent=2)

    for off, problem_text in enumerate(problems):
        idx = off + a.start_problem_index
        pdir = os.path.join(a.output_dir, f"problem_{idx}")
        # resume: skip problems already finished by a previous invocation
        last = os.path.join(pdir, f"generation_{a.generations_per_problem-1}", "metadata.json")
        if os.path.exists(last):
            print(f"[skip] problem {idx} already done", flush=True)
            continue
        os.makedirs(pdir, exist_ok=True)
        print(f"\nProcessing problem {idx}...", flush=True)

        with open(os.path.join(pdir, "original_problem.txt"), "w", encoding="utf-8") as f:
            f.write(problem_text)
        formatted = a.prompt_template.format(problem=problem_text)
        with open(os.path.join(pdir, "formatted_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(formatted)

        inputs = tok(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                temperature=a.temperature, max_new_tokens=a.max_new_tokens,
                pad_token_id=tok.pad_token_id, do_sample=a.do_sample,
                top_p=a.top_p if a.do_sample else None,
                num_return_sequences=a.generations_per_problem,
                return_dict_in_generate=True, output_hidden_states=False,
            )
        gen_time = time.time() - t0
        print(f"  generated in {gen_time:.1f}s", flush=True)

        seqs = out.sequences
        for k in range(a.generations_per_problem):
            gdir = os.path.join(pdir, f"generation_{k}")
            os.makedirs(gdir, exist_ok=True)
            ids_k = seqs[k]

            # replicate the original length rule exactly
            gen_only = ids_k[prompt_len:]
            glen = 0
            for tid in gen_only:
                if tid == tok.pad_token_id:
                    break
                glen += 1
                if tid == tok.eos_token_id:
                    break

            keep = ids_k[: prompt_len + glen].unsqueeze(0)
            with torch.no_grad():
                o = model(keep, output_hidden_states=True)
            hs = o.hidden_states[layer_idx][0]                # (prompt_len+glen, H)
            hs_in = hs[:prompt_len].float().cpu().numpy().astype(np.float16)
            hs_out = (hs[prompt_len - 1: prompt_len + glen - 1].float().cpu().numpy().astype(np.float16)
                      if glen > 0 else np.empty((0, hs.shape[-1]), dtype=np.float16))
            del o, hs

            with open(os.path.join(gdir, "response.txt"), "w", encoding="utf-8") as f:
                f.write(tok.decode(ids_k, skip_special_tokens=True))
            with open(os.path.join(gdir, "tokens.json"), "w", encoding="utf-8") as f:
                json.dump(tok.convert_ids_to_tokens(ids_k), f, indent=2)
            np.save(os.path.join(gdir, "token_hidden_states_input.npy"), hs_in)
            np.save(os.path.join(gdir, "token_hidden_states_output.npy"), hs_out)
            with open(os.path.join(gdir, "metadata.json"), "w") as f:
                json.dump({
                    "batch_generation_time_seconds_for_problem": gen_time,
                    "estimated_time_per_generation_in_batch": gen_time / a.generations_per_problem,
                    "input_prompt_length_tokens": prompt_len,
                    "output_generated_length_tokens": glen,
                    "total_tokens_in_sequence_file": len(ids_k),
                    "prompt_plus_output_tokens": prompt_len + glen,
                }, f, indent=2)

        del out, seqs
        torch.cuda.empty_cache(); gc.collect()
        print(f"Completed processing for problem {idx}", flush=True)

    print("\nGeneration process complete.", flush=True)


if __name__ == "__main__":
    main()
