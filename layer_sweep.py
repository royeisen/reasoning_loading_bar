"""Which layer carries the thinking-progress signal?

generate_tpv_data.py hardcodes the LAST layer (hidden_states[...][-1]). In probing
work the final layer is often not the most linearly decodable, since it is
specialised for next-token prediction. This re-runs forward passes over the
ALREADY-GENERATED sequences (no regeneration) and fits the same closed-form TPV
probe at several representative depths.

Rather than materialising hidden states, we stream the regression sufficient
statistics per layer:
    A = H^T H    b = H^T y    yty = y^T y    ysum, n
which gives the exact closed-form solution and exact R^2:
    SS_res = yty - 2 w.b + w^T A w        SS_tot = yty - ysum^2/n
Memory is 9 layers x 4096^2 float64 = ~1.2 GB regardless of dataset size.

Train/test split mirrors prepare_tpv_dataset.py exactly.
"""
import argparse, glob, json, os, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Stats:
    def __init__(self, d, dev):
        self.A = torch.zeros(d, d, dtype=torch.float64, device=dev)
        self.b = torch.zeros(d, 1, dtype=torch.float64, device=dev)
        self.yty = 0.0
        self.ysum = 0.0
        self.n = 0

    def add(self, H, y):
        Hf = H.float()
        self.A += (Hf.T @ Hf).double()
        self.b += (Hf.T @ y.float()).double()
        self.yty += float((y * y).sum())
        self.ysum += float(y.sum())
        self.n += y.shape[0]


def solve_and_score(tr, te, lam):
    d = tr.A.shape[0]
    w = torch.linalg.solve(tr.A + lam * torch.eye(d, dtype=torch.float64, device=tr.A.device), tr.b)
    out = []
    for s in (tr, te):
        ss_res = s.yty - 2.0 * float((w * s.b).sum()) + float((w.T @ s.A @ w).squeeze())
        ss_tot = s.yty - (s.ysum ** 2) / s.n
        out.append(1.0 - ss_res / ss_tot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="", help="comma list; default = 9 evenly spaced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_split_ratio", type=float, default=0.8)
    ap.add_argument("--end_of_thinking_token", default="</think>")
    ap.add_argument("--lam", type=float, default=1e-10)
    ap.add_argument("--max_gen_per_problem", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map="cuda")
    model.eval()
    n_hidden = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    layers = ([int(x) for x in a.layers.split(",")] if a.layers
              else sorted(set(np.linspace(0, n_hidden - 1, 9).astype(int).tolist())))
    print(f"model has {n_hidden} hidden-state entries (dim {d}); probing layers {layers}", flush=True)

    # ---- collect usable generations, grouped by problem (mirrors prepare_tpv_dataset) ----
    per_problem = {}
    for pdir in sorted(glob.glob(os.path.join(a.run_dir, "problem_*"))):
        try:
            pid = int(os.path.basename(pdir).split("_")[-1])
        except ValueError:
            continue
        items = []
        for gdir in sorted(glob.glob(os.path.join(pdir, "generation_*"))):
            mp, tp = os.path.join(gdir, "metadata.json"), os.path.join(gdir, "tokens.json")
            if os.path.exists(os.path.join(gdir, "error.txt")) or not (os.path.exists(mp) and os.path.exists(tp)):
                continue
            md, toks = json.load(open(mp)), json.load(open(tp))
            pl, gl = md["input_prompt_length_tokens"], md["output_generated_length_tokens"]
            gen = toks[pl:pl + gl]
            if a.end_of_thinking_token not in gen:
                continue
            N = gen.index(a.end_of_thinking_token) + 1
            items.append((toks[:pl + N], pl, N))
        if items:
            per_problem[pid] = items

    pids = list(per_problem.keys())
    random.seed(a.seed); random.shuffle(pids)
    n_tr = int(len(pids) * a.train_split_ratio)
    split = {p: ("train" if i < n_tr else "test") for i, p in enumerate(pids)}
    print(f"problems {len(pids)} -> train {n_tr} / test {len(pids)-n_tr}", flush=True)

    dev = "cuda"
    stats = {L: {"train": Stats(d, dev), "test": Stats(d, dev)} for L in layers}

    done = 0
    for pid, items in per_problem.items():
        sp = split[pid]
        use = items if a.max_gen_per_problem <= 0 else items[: a.max_gen_per_problem]
        for toks, pl, N in use:
            ids = torch.tensor([tok.convert_tokens_to_ids(toks)], device=dev)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            y = (torch.arange(1, N + 1, device=dev, dtype=torch.float64) / N).reshape(-1, 1)
            for L in layers:
                H = out.hidden_states[L][0][pl - 1: pl + N - 1]
                stats[L][sp].add(H, y)
            del out
            done += 1
            if done % 50 == 0:
                print(f"  {done} generations processed", flush=True)
        torch.cuda.empty_cache()

    print(f"\nprocessed {done} generations\n", flush=True)
    print(f"{'layer':>6}  {'train R2':>9}  {'test R2':>9}")
    best = None
    for L in layers:
        tr, te = solve_and_score(stats[L]["train"], stats[L]["test"], a.lam)
        tag = ""
        if best is None or te > best[1]:
            best = (L, te); tag = ""
        print(f"{L:>6}  {tr:>9.4f}  {te:>9.4f}{tag}")
    print(f"\nbest layer by test R2: {best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    main()
