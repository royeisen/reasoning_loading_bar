"""R^2 for the TPV regressor, reusing train_tpv.py's own closed-form fit.

train_tpv.py reports MSE only. This adds R^2, and optionally the within-generation
breakdown: pooled R^2 over a flat set of (hidden_state, i/N) pairs can flatter a probe
that merely knows roughly how far into an average trace a state sits, whereas the
monitoring claim is about tracking progress inside a single generation.

Kept separate from train_tpv.py so the reproduction path stays untouched.

  --input_dir   a prepared dataset dir (train/test .pkl) -> pooled R^2
  --run_dir     a raw hidden-states dir                  -> adds per-generation R^2
                (needs the raw dir because the .pkl files drop generation boundaries)
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch

from train_tpv import load_datasets, prepare_tensors, fit_linear_regression, predict


def r2_mse(pred, y):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            float(((y - pred) ** 2).mean()))


def collect_generations(run_dir, end_tok="</think>"):
    """{problem_id: [(H, y), ...]} with one array pair per usable generation."""
    per_problem = {}
    for pdir in sorted(glob.glob(os.path.join(run_dir, "problem_*"))):
        if not os.path.isdir(pdir):
            continue
        try:
            pid = int(os.path.basename(pdir).split("_")[-1])
        except ValueError:
            continue
        gens = []
        for gdir in sorted(glob.glob(os.path.join(pdir, "generation_*"))):
            mp = os.path.join(gdir, "metadata.json")
            tp = os.path.join(gdir, "tokens.json")
            hp = os.path.join(gdir, "token_hidden_states_output.npy")
            if os.path.exists(os.path.join(gdir, "error.txt")):
                continue
            if not all(os.path.exists(p) for p in (mp, tp, hp)):
                continue
            md, toks = json.load(open(mp)), json.load(open(tp))
            pl, gl = md["input_prompt_length_tokens"], md["output_generated_length_tokens"]
            hs = np.load(hp)
            if hs.shape[0] != gl:
                continue
            gen_toks = toks[pl:pl + gl]
            if end_tok not in gen_toks:
                continue
            N = gen_toks.index(end_tok) + 1
            gens.append((hs[:N], np.arange(1, N + 1, dtype=np.float64) / N))
        if gens:
            per_problem[pid] = gens
    return per_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", help="prepared dataset dir (train/test .pkl)")
    ap.add_argument("--run_dir", help="raw hidden-states dir; enables --per_generation")
    ap.add_argument("--per_generation", action="store_true",
                    help="also report R^2 computed inside each held-out generation")
    ap.add_argument("--epsilon", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_split_ratio", type=float, default=0.8)
    ap.add_argument("--end_of_thinking_token", default="</think>")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if not a.input_dir and not a.run_dir:
        ap.error("need --input_dir or --run_dir")
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    if a.input_dir:
        train, test = load_datasets(a.input_dir)
        H_tr, y_tr = prepare_tensors(train, dev)
        H_te, y_te = prepare_tensors(test, dev)
        print(f"train records {len(train)}  test records {len(test)}  d={H_tr.shape[1]}")
        w = fit_linear_regression(H_tr, y_tr, epsilon=a.epsilon)
        r2_tr, mse_tr = r2_mse(predict(H_tr, w).cpu(), y_tr.cpu())
        r2_te, mse_te = r2_mse(predict(H_te, w).cpu(), y_te.cpu())
        print(f"  train  R^2 = {r2_tr: .4f}   MSE = {mse_tr:.6f}")
        print(f"  test   R^2 = {r2_te: .4f}   MSE = {mse_te:.6f}")

    if a.per_generation:
        if not a.run_dir:
            ap.error("--per_generation requires --run_dir (the .pkl files drop "
                     "generation boundaries)")
        per_problem = collect_generations(a.run_dir, a.end_of_thinking_token)
        if not per_problem:
            print("no usable generations")
            return
        # mirror prepare_tpv_dataset.py's split exactly
        pids = list(per_problem.keys())
        random.seed(a.seed)
        random.shuffle(pids)
        n_tr = int(len(pids) * a.train_split_ratio)
        train_ids, test_ids = pids[:n_tr], pids[n_tr:]

        def stack(ids):
            H = np.concatenate([g[0] for p in ids for g in per_problem[p]], axis=0)
            y = np.concatenate([g[1] for p in ids for g in per_problem[p]], axis=0)
            return H, y

        H_tr, y_tr = stack(train_ids)
        w = fit_linear_regression(
            torch.tensor(H_tr, dtype=torch.float64, device=dev),
            torch.tensor(y_tr, dtype=torch.float64, device=dev).reshape(-1, 1),
            epsilon=a.epsilon,
        ).cpu().numpy()

        H_te, y_te = stack(test_ids)
        pooled, _ = r2_mse(H_te.astype(np.float64) @ w, y_te)
        vals = [r2_mse(Hg.astype(np.float64) @ w, yg)[0]
                for p in test_ids for Hg, yg in per_problem[p]]
        per_gen = np.array([v for v in vals if np.isfinite(v)])

        print(f"\nproblems {len(pids)}  train {len(train_ids)}  test {len(test_ids)}"
              f"  test generations {len(per_gen)}")
        print(f"pooled test R^2        = {pooled: .4f}")
        print(f"per-generation test R^2: mean {per_gen.mean(): .4f}  "
              f"median {np.median(per_gen): .4f}  "
              f"p10 {np.percentile(per_gen, 10): .4f}  "
              f"p90 {np.percentile(per_gen, 90): .4f}")
        print(f"  generations with R^2 < 0: {(per_gen < 0).sum()}/{len(per_gen)}")


if __name__ == "__main__":
    main()
