"""Per-generation R^2 for the TPV regressor.

prepare_tpv_dataset.py flattens everything into (hidden_state, i/N) pairs, losing
generation boundaries. Pooled R^2 over that flat set is partly inflated: a model can
score well just by knowing roughly how far into an average trace a state is.
The monitoring claim is about tracking progress WITHIN one generation, so we also
report R^2 computed inside each test generation separately.

The train/test problem split here mirrors prepare_tpv_dataset.py exactly
(sorted glob order -> random.seed(seed) -> shuffle -> 80/20) so the held-out set
is identical to the one train_tpv.py never saw.
"""
import argparse, glob, json, os, random
import numpy as np
import torch

from train_tpv import fit_linear_regression


def collect(input_dir, end_tok="</think>"):
    """Return {problem_id: [(H, y), ...]} with one (H, y) array pair per generation."""
    per_problem = {}
    for pdir in sorted(glob.glob(os.path.join(input_dir, "problem_*"))):
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
            md = json.load(open(mp))
            toks = json.load(open(tp))
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


def r2(pred, y):
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="hidden states dir (generate_tpv_data output)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_split_ratio", type=float, default=0.8)
    ap.add_argument("--epsilon", type=float, default=1e-10)
    ap.add_argument("--end_of_thinking_token", default="</think>")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    per_problem = collect(a.input_dir, a.end_of_thinking_token)
    if not per_problem:
        print("no usable generations")
        return

    pids = list(per_problem.keys())
    random.seed(a.seed)
    random.shuffle(pids)
    n_tr = int(len(pids) * a.train_split_ratio)
    train_ids, test_ids = pids[:n_tr], pids[n_tr:]
    print(f"problems with data: {len(pids)}  train {len(train_ids)}  test {len(test_ids)}")
    print(f"test generations: {sum(len(per_problem[p]) for p in test_ids)}")

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

    # pooled over all held-out token positions
    H_te, y_te = stack(test_ids)
    pooled = r2(H_te.astype(np.float64) @ w, y_te)

    # within each held-out generation
    vals = []
    for p in test_ids:
        for Hg, yg in per_problem[p]:
            vals.append(r2(Hg.astype(np.float64) @ w, yg))
    per_gen = np.array([v for v in vals if np.isfinite(v)])

    print(f"\npooled test R^2        = {pooled: .4f}   (comparable to train_tpv.py MSE)")
    print(f"per-generation test R^2: mean {per_gen.mean(): .4f}  median {np.median(per_gen): .4f}"
          f"  p10 {np.percentile(per_gen, 10): .4f}  p90 {np.percentile(per_gen, 90): .4f}")
    print(f"  generations with R^2 < 0: {(per_gen < 0).sum()}/{len(per_gen)}")


if __name__ == "__main__":
    main()
