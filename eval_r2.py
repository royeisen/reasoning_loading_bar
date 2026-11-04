"""Report R^2 for the TPV regressor, reusing train_tpv.py's exact closed-form fit.

Kept separate from train_tpv.py so the reproduction path stays untouched.
"""
import argparse, os, pickle, glob, json
import numpy as np
import torch

from train_tpv import load_datasets, prepare_tensors, fit_linear_regression, predict


def r2(pred, y):
    pred = pred.detach().cpu().numpy().ravel().astype(np.float64)
    y = y.detach().cpu().numpy().ravel().astype(np.float64)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot, float(((y - pred) ** 2).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--epsilon", type=float, default=1e-10)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    train, test = load_datasets(a.input_dir)
    H_tr, y_tr = prepare_tensors(train, dev)
    H_te, y_te = prepare_tensors(test, dev)
    print(f"train records {len(train)}  test records {len(test)}  d={H_tr.shape[1]}")

    # float32 normal equations are ill-conditioned at d=4096; fit in float64.
    w32 = fit_linear_regression(H_tr, y_tr, epsilon=a.epsilon)
    w64 = fit_linear_regression(H_tr.double(), y_tr.double(), epsilon=a.epsilon)

    for name, w, Ht, Hv in (("float32 (repo default)", w32, H_tr, H_te),
                            ("float64", w64, H_tr.double(), H_te.double())):
        r2_tr, mse_tr = r2(predict(Ht, w), y_tr)
        r2_te, mse_te = r2(predict(Hv, w), y_te)
        print(f"\n[{name}]")
        print(f"  train  R^2 = {r2_tr: .4f}   MSE = {mse_tr:.6f}")
        print(f"  test   R^2 = {r2_te: .4f}   MSE = {mse_te:.6f}")



    # Diagnostic: the repo fit has no intercept. Hidden states carry a large
    # near-constant component so an implicit bias exists, but check explicitly.
    def with_bias(H):
        return torch.cat([H, torch.ones(H.shape[0], 1, dtype=H.dtype, device=H.device)], dim=1)
    Hb_tr, Hb_te = with_bias(H_tr.double()), with_bias(H_te.double())
    wb = fit_linear_regression(Hb_tr, y_tr.double(), epsilon=a.epsilon)
    r2_tr_b, _ = r2(predict(Hb_tr, wb), y_tr)
    r2_te_b, _ = r2(predict(Hb_te, wb), y_te)
    print(f"\n[float64 + intercept (diagnostic, not the repo model)]")
    print(f"  train  R^2 = {r2_tr_b: .4f}")
    print(f"  test   R^2 = {r2_te_b: .4f}")


if __name__ == "__main__":
    main()
