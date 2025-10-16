#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, random, argparse, platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, Matern
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm

np.random.seed(42); random.seed(42)

WD       = "Radiosensitivity Prediction"
DATA_DIR = f"{WD}/src/data/Cleveland"
X_PATH   = Path(DATA_DIR) / "transcriptomics_imputed.csv"
Y_PATH   = Path(DATA_DIR) / "cleveland_auc_only.csv"

OUT_ROOT = Path(f"{WD}/gpr_nll")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def timestamp(): return datetime.now().strftime("%Y%m%d_%H%M%S")

# Toggles
FAST = os.getenv("FAST","0") == "1"
MEAN_GPR_RESTARTS = 1 if FAST else 3
VAR_GPR_RESTARTS  = 1 if FAST else 2
USE_C_HOMO        = (os.getenv("USE_C_HOMO","1") == "1")  

# AUC_BINS = np.array([-np.inf, 2.0, 3.0, 6.0, np.inf])
# AUC_BIN_NAMES = [
#     "Highly Sensitive (≤2)",
#     "Bulk (2–3)",
#     "Moderate (3–6)",
#     "Highly Resistant (>6)",
# ]
# def auc_bin_index(mu): return np.digitize(mu, AUC_BINS) - 1
# def auc_bin_name_from_idx(idx): return AUC_BIN_NAMES[int(idx)]

def median_lengthscale(X, max_n=800, rng=42):
    """Median pairwise distance on a subsample (NO PCA) → ell. X must be standardized."""
    rs = np.random.RandomState(rng)
    n = X.shape[0]
    idx = rs.choice(n, size=min(max_n, n), replace=False)
    Z = X[idx]
    m = min(400, Z.shape[0])
    Z = Z[:m]
    d2 = np.sum((Z[:, None, :] - Z[None, :, :])**2, axis=-1)
    d = np.sqrt(d2[np.triu_indices(m, k=1)])
    med = np.median(d)
    return float(max(med, 1e-6))

def build_kernel(tag, ell):
    b = (ell/100.0, ell*100.0)
    if tag == "rbf_rq_white":
        return (RBF(length_scale=ell, length_scale_bounds=b)
                + RationalQuadratic(length_scale=ell, alpha=1.0,
                                    length_scale_bounds=b, alpha_bounds=(1e-3,1e5))
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6,10.0)))
    raise ValueError(tag)

def extract_white_noise_level(k):
    from sklearn.gaussian_process.kernels import WhiteKernel
    if hasattr(k, 'k1') and hasattr(k, 'k2'):
        return extract_white_noise_level(k.k1) + extract_white_noise_level(k.k2)
    if isinstance(k, WhiteKernel):
        return float(k.noise_level)
    return 0.0

def nll_gaussian(y, mu, var, eps=1e-12):
    v = np.maximum(var, eps); r = y - mu
    return float(np.mean(0.5*np.log(2*np.pi*v) + 0.5*(r*r)/v))

def coverage(y, mu, sigma, k=1.0):
    return float(np.mean(np.abs(y-mu) <= k*sigma))

def reliability_points(y, mu, sigma):
    levels = np.array([0.50, 0.6827, 0.80, 0.90, 0.95])
    ks = norm.ppf(0.5*(1+levels))
    cov = [coverage(y, mu, sigma, k) for k in ks]
    return levels, cov

def nll_mean_from_res2_and_var(res2, var, eps=1e-12):
    v = np.maximum(var, eps)
    return float(np.mean(0.5*np.log(2*np.pi*v) + 0.5*res2 / v))

def argmin_grid(func, grid):
    vals = [func(x) for x in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), float(vals[i])

def plot_pred_vs_obs_with_bars(all_test, title, outdir):
    y  = all_test["y_true"].values
    yp = all_test["y_pred"].values
    st = all_test["sigma_total"].values
    plt.figure(figsize=(8.2,6.4))
    plt.errorbar(y, yp, yerr=2*st, fmt='o', ms=3.5, elinewidth=0.6, alpha=0.20)
    plt.scatter(y, yp, s=12, alpha=0.55, edgecolor='k', linewidth=0.2)
    mn, mx = float(min(y.min(), yp.min())), float(max(y.max(), yp.max()))
    plt.plot([mn,mx],[mn,mx],'k--',lw=1.4,alpha=0.8,label='Perfect prediction')
    plt.axhline(y=np.mean(y), color='k', ls=':', lw=1.0, label='Mean baseline')
    plt.xlabel("Observed AUC"); plt.ylabel("Predicted AUC")
    plt.title(title)
    plt.legend(loc='upper left'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir/"pred_vs_obs.png", dpi=300, bbox_inches='tight')
    plt.savefig(outdir/"pred_vs_obs.pdf", bbox_inches='tight')
    plt.close()

def plot_reliability(all_test, title, outdir):
    y, yp, st = all_test["y_true"].values, all_test["y_pred"].values, all_test["sigma_total"].values
    levels, cov = reliability_points(y, yp, st)
    plt.figure(figsize=(6.4,5.2))
    plt.plot(levels, levels, 'k--', lw=1.2, label='Ideal')
    plt.plot(levels, cov, 'o-', label='Empirical')
    plt.xlim(0.45, 0.97); plt.ylim(0.45, 0.97)
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"reliability.png", dpi=300, bbox_inches='tight')
    plt.savefig(outdir/"reliability.pdf", bbox_inches='tight')
    plt.close()

def plot_fold_grid(train_rows, test_rows, title, outdir):
    fig, axes = plt.subplots(2, 5, figsize=(18,7), sharex=True, sharey=True)
    for i in range(5):
        for r, split, rows in [(0,'TEST',test_rows), (1,'TRAIN',train_rows)]:
            ax = axes[r,i]
            df = rows[rows["outer_fold"]==i+1]
            if len(df)==0: continue
            y, yp = df["y_true"].values, df["y_pred"].values
            st = df["sigma_total"].values
            ax.errorbar(y, yp, yerr=2*st, fmt='o', ms=3, elinewidth=0.5, alpha=0.20)
            ax.scatter(y, yp, s=10, alpha=0.55, edgecolor='k', linewidth=0.2)
            mn, mx = float(min(y.min(), yp.min())), float(max(y.max(), yp.max()))
            ax.plot([mn,mx],[mn,mx],'k--',lw=1.0,alpha=0.8)
            r2 = r2_score(y, yp); nll = nll_gaussian(y, yp, st**2)
            ax.set_title(f"Fold {i+1} — {split}\nR²={r2:.3f}, NLL={nll:.3f}", fontsize=10)
            ax.grid(alpha=0.25)
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(outdir/"foldwise_panels.png", dpi=300, bbox_inches='tight')
    plt.savefig(outdir/"foldwise_panels.pdf", bbox_inches='tight')
    plt.close(fig)

def self_check_and_save(train_df, test_df, outdir, rtol=1e-6, atol=1e-9):
    def check(df):
        lhs = np.asarray(df["sigma_total"])**2
        rhs = np.asarray(df["sigma_epistemic"])**2 + np.asarray(df["sigma_aleatoric"])**2
        ok  = np.allclose(lhs, rhs, rtol=rtol, atol=atol)
        max_abs_err = float(np.max(np.abs(lhs - rhs))) if len(df) else 0.0
        return dict(ok=bool(ok), max_abs_err=max_abs_err, n=len(df))
    report = {"train": check(train_df), "test": check(test_df), "rtol": rtol, "atol": atol}
    with open(outdir / "selfcheck.json", "w") as f: json.dump(report, f, indent=2)
    print(f"[selfcheck] train ok={report['train']['ok']} (max_abs_err={report['train']['max_abs_err']:.3e}), "
          f"test ok={report['test']['ok']} (max_abs_err={report['test']['max_abs_err']:.3e})")

def fit_mean_gp(Xtr, ytr, kernel, alpha=None, restarts=3):
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        alpha=alpha if alpha is not None else 1e-10,
        n_restarts_optimizer=restarts, random_state=0
    )
    gpr.fit(Xtr, ytr)
    return gpr

def evaluate_branch(save_dir, tag, branch_name, y_te, mu_te, var_epi_te, sigma_ale_te):
    sig_te = np.sqrt(np.maximum(var_epi_te + sigma_ale_te, 1e-12))
    r2  = r2_score(y_te, mu_te)
    mse = mean_squared_error(y_te, mu_te)
    nll = nll_gaussian(y_te, mu_te, sig_te**2)
    cov1 = coverage(y_te, mu_te, sig_te, k=1.0)
    cov2 = coverage(y_te, mu_te, sig_te, k=2.0)
    metrics = dict(R2=r2, MSE=mse, NLL=nll, cov1=cov1, cov2=cov2)

    # Save metrics for this branch
    with open(save_dir / f"{branch_name}_metrics.json","w") as f: json.dump(metrics, f, indent=2)

    return r2, mse, nll, cov1, cov2

def run_model(tag, X, y, outer_cv, inner_cv, run_dir):
    model_dir = run_dir / tag
    (model_dir/"meta").mkdir(parents=True, exist_ok=True)

    cfg = dict(model=tag, outer_folds=5, inner_folds=5, random_seed=42,
               fast=FAST, use_c_homo=USE_C_HOMO)
    with open(model_dir/"meta"/"config.json","w") as f: json.dump(cfg, f, indent=2)
    vers = {"python": sys.version, "platform": platform.platform()}
    try:
        import numpy, pandas, sklearn
        vers.update({"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__})
    except Exception:
        pass
    with open(model_dir/"meta"/"versions.json","w") as f: json.dump(vers, f, indent=2)

    results_hetero, results_homo = [], []
    rows_test_hetero, rows_train_hetero = [], []
    rows_test_homo,  rows_train_homo  = [], []
    t0 = time.time()

    for ofold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        fold_dir = model_dir / f"fold_{ofold:02d}"
        (fold_dir/"hetero").mkdir(parents=True, exist_ok=True)
        (fold_dir/"homo").mkdir(parents=True, exist_ok=True)

        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr,     y_te     = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        def fit_get_nll(Xtr, ytr, Xval, yval):
            ell = median_lengthscale(Xtr, rng=42+ofold)
            kern = build_kernel(tag, ell)
            gpr  = fit_mean_gp(Xtr, ytr, kernel=kern, alpha=None, restarts=MEAN_GPR_RESTARTS)
            mu, std = gpr.predict(Xval, return_std=True)
            return nll_gaussian(yval, mu, std**2), gpr.kernel_

        fold_nlls, best_kernels = [], []
        for itr, ival in inner_cv.split(X_tr, y_tr):
            Xitr, Xval = X_tr[itr], X_tr[ival]
            yitr, yval = y_tr[itr], y_tr[ival]
            nll, ker = fit_get_nll(Xitr, yitr, Xval, yval)
            fold_nlls.append(nll); best_kernels.append(ker)

        fold_nlls_arr = np.array(fold_nlls, dtype=float)
        avg_inner_nll = float(fold_nlls_arr.mean())
        closest_idx = int(np.argmin(np.abs(fold_nlls_arr - avg_inner_nll)))
        chosen_kernel = best_kernels[closest_idx]
        print(f"[{tag}] Outer {ofold}: inner NLL mean={avg_inner_nll:.4f}; using kernel: {chosen_kernel}")

        gpr_mean = fit_mean_gp(X_tr, y_tr, kernel=chosen_kernel, alpha=None, restarts=MEAN_GPR_RESTARTS)
        mu_tr, std_tr = gpr_mean.predict(X_tr, return_std=True)
        mu_te, std_te = gpr_mean.predict(X_te, return_std=True)
        sigma_white   = extract_white_noise_level(gpr_mean.kernel_)  # variance from WhiteKernel
        var_epi_tr = np.maximum(std_tr**2 - sigma_white, 1e-12)
        var_epi_te = np.maximum(std_te**2 - sigma_white, 1e-12)

        subcv = KFold(n_splits=5, shuffle=True, random_state=99)
        oof_idx, oof_logres2, oof_res2, oof_var_epi, oof_res_signed = [], [], [], [], []
        for s_tr, s_val in subcv.split(X_tr, y_tr):
            Xs_tr, Xs_val = X_tr[s_tr], X_tr[s_val]
            ys_tr, ys_val = y_tr[s_tr], y_tr[s_val]
            gpr_sub = fit_mean_gp(Xs_tr, ys_tr, kernel=gpr_mean.kernel_, alpha=None, restarts=1)
            muv, stdv = gpr_sub.predict(Xs_val, return_std=True)
            res = ys_val - muv
            z = np.log(np.maximum(res**2, 1e-12))
            oof_idx.append(s_val); oof_logres2.append(z)
            oof_res2.append(res**2); oof_res_signed.append(res)
            sigma_white_sub = extract_white_noise_level(gpr_sub.kernel_)
            oof_var_epi.append(np.maximum(stdv**2 - sigma_white_sub, 1e-12))

        oof_idx       = np.concatenate(oof_idx)
        oof_logres2   = np.concatenate(oof_logres2)
        oof_res2      = np.concatenate(oof_res2)
        oof_var_epi   = np.concatenate(oof_var_epi)
        oof_res_signed= np.concatenate(oof_res_signed)
        X_oof         = X_tr[oof_idx]

        ell_var = median_lengthscale(X_tr, rng=7+ofold)
        k_var   = RBF(length_scale=ell_var, length_scale_bounds=(ell_var/100, ell_var*100)) \
                + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
        gpr_var = GaussianProcessRegressor(kernel=k_var, normalize_y=False,
                                           n_restarts_optimizer=VAR_GPR_RESTARTS, random_state=0)
        gpr_var.fit(X_oof, oof_logres2)

        def predict_var_alea(Xmat):
            mu_z, std_z = gpr_var.predict(Xmat, return_std=True)
            return np.exp(mu_z + 0.5*std_z**2)  # E[exp(Z)] for Z~N
        var_alea_tr = predict_var_alea(X_tr)
        var_alea_te = predict_var_alea(X_te)
        var_alea_oof= predict_var_alea(X_oof)

        mu_oof, _ = gpr_mean.predict(X_oof, return_std=True)
        bin_idx_oof = auc_bin_index(mu_oof)

        c_grid = np.geomspace(1e-3, 3.0, 61)
        def f_nll_c(cand):
            return nll_mean_from_res2_and_var(oof_res2, oof_var_epi + cand * var_alea_oof)
        c, _ = argmin_grid(f_nll_c, c_grid)

        a_bins = np.zeros(len(AUC_BIN_NAMES), dtype=float)
        for b in range(len(AUC_BIN_NAMES)):
            m = (bin_idx_oof == b)
            if not np.any(m):
                a_bins[b] = 0.0
                continue
            base_var_b = oof_var_epi[m] + c * var_alea_oof[m]
            res2_b = oof_res2[m]
            cap = max(1e-8, float(np.quantile(res2_b, 0.9)))
            a_grid = np.linspace(0.0, cap, 51)
            def f_nll_a(a):
                return nll_mean_from_res2_and_var(res2_b, base_var_b + a)
            a_opt, _ = argmin_grid(f_nll_a, a_grid)
            a_bins[b] = max(0.0, a_opt)

        bin_idx_tr = auc_bin_index(mu_tr)
        bin_idx_te = auc_bin_index(mu_te)
        var_tot_tr_hetero = var_epi_tr + c * var_alea_tr + a_bins[bin_idx_tr]
        var_tot_te_hetero = var_epi_te + c * var_alea_te + a_bins[bin_idx_te]


        if USE_C_HOMO:
            # OOF residuals already computed above; build constant baseline
            sigma_white_oof = np.full_like(oof_var_epi, fill_value=sigma_white)
            def f_nll_c_homo(cand):
                return nll_mean_from_res2_and_var(oof_res2, oof_var_epi + cand * sigma_white_oof)
            c_homo, _ = argmin_grid(f_nll_c_homo, c_grid)
        else:
            c_homo = 1.0

        var_tot_tr_homo = var_epi_tr + c_homo * sigma_white
        var_tot_te_homo = var_epi_te + c_homo * sigma_white

        sig_tr_he = np.sqrt(np.maximum(var_tot_tr_hetero, 1e-12))
        sig_te_he = np.sqrt(np.maximum(var_tot_te_hetero, 1e-12))
        r2_he  = r2_score(y_te, mu_te)
        mse_he = mean_squared_error(y_te, mu_te)
        nll_he = nll_gaussian(y_te, mu_te, sig_te_he**2)
        cov1_he= coverage(y_te, mu_te, sig_te_he, k=1.0)
        cov2_he= coverage(y_te, mu_te, sig_te_he, k=2.0)

        for i, idx in enumerate(te_idx):
            rows_test_hetero.append({
                "outer_fold": ofold, "split": "test", "row_index": int(idx),
                "y_true": float(y[idx]), "y_pred": float(mu_te[i]),
                "auc_bin": auc_bin_name_from_idx(bin_idx_te[i]),
                "sigma_epistemic": float(np.sqrt(var_epi_te[i])),
                "sigma_aleatoric": float(np.sqrt(max(var_tot_te_hetero[i] - var_epi_te[i], 1e-12))),
                "sigma_total": float(sig_te_he[i])
            })
        for i, idx in enumerate(tr_idx):
            rows_train_hetero.append({
                "outer_fold": ofold, "split": "train", "row_index": int(idx),
                "y_true": float(y[idx]), "y_pred": float(mu_tr[i]),
                "auc_bin": auc_bin_name_from_idx(bin_idx_tr[i]),
                "sigma_epistemic": float(np.sqrt(var_epi_tr[i])),
                "sigma_aleatoric": float(np.sqrt(max(var_tot_tr_hetero[i] - var_epi_tr[i], 1e-12))),
                "sigma_total": float(sig_tr_he[i])
            })

        results_hetero.append({
            "outer_fold": ofold, "R2": r2_he, "MSE": mse_he, "NLL": nll_he,
            "cov@1sigma": cov1_he, "cov@2sigma": cov2_he,
            "OOF_cov@1sigma": float(np.mean(np.abs(oof_res_signed) <= np.sqrt(np.maximum(oof_var_epi + c*var_alea_oof + a_bins[bin_idx_oof],1e-12)))),
            "sigma_white(var)": sigma_white, "var_calib_c": c,
            "a_bin_sensitive": float(a_bins[0]), "a_bin_bulk": float(a_bins[1]),
            "a_bin_moderate": float(a_bins[2]), "a_bin_resistant": float(a_bins[3])
        })

        sig_tr_ho = np.sqrt(np.maximum(var_tot_tr_homo, 1e-12))
        sig_te_ho = np.sqrt(np.maximum(var_tot_te_homo, 1e-12))
        r2_ho  = r2_score(y_te, mu_te)          # identical to hetero (same mean)
        mse_ho = mean_squared_error(y_te, mu_te) # identical to hetero (same mean)
        nll_ho = nll_gaussian(y_te, mu_te, sig_te_ho**2)
        cov1_ho= coverage(y_te, mu_te, sig_te_ho, k=1.0)
        cov2_ho= coverage(y_te, mu_te, sig_te_ho, k=2.0)

        for i, idx in enumerate(te_idx):
            rows_test_homo.append({
                "outer_fold": ofold, "split": "test", "row_index": int(idx),
                "y_true": float(y[idx]), "y_pred": float(mu_te[i]),
                "auc_bin": auc_bin_name_from_idx(bin_idx_te[i]),
                "sigma_epistemic": float(np.sqrt(var_epi_te[i])),
                "sigma_aleatoric": float(np.sqrt(max(var_tot_te_homo[i] - var_epi_te[i], 1e-12))),
                "sigma_total": float(sig_te_ho[i])
            })
        for i, idx in enumerate(tr_idx):
            rows_train_homo.append({
                "outer_fold": ofold, "split": "train", "row_index": int(idx),
                "y_true": float(y[idx]), "y_pred": float(mu_tr[i]),
                "auc_bin": auc_bin_name_from_idx(bin_idx_tr[i]),
                "sigma_epistemic": float(np.sqrt(var_epi_tr[i])),
                "sigma_aleatoric": float(np.sqrt(max(var_tot_tr_homo[i] - var_epi_tr[i], 1e-12))),
                "sigma_total": float(sig_tr_ho[i])
            })

        results_homo.append({
            "outer_fold": ofold, "R2": r2_ho, "MSE": mse_ho, "NLL": nll_ho,
            "cov@1sigma": cov1_ho, "cov@2sigma": cov2_ho,
            "sigma_white(var)": sigma_white, "c_homo": float(c_homo)
        })

        pd.DataFrame(rows_test_hetero).query("outer_fold == @ofold").to_csv(fold_dir/"hetero/test_preds.csv", index=False)
        pd.DataFrame(rows_train_hetero).query("outer_fold == @ofold").to_csv(fold_dir/"hetero/train_preds.csv", index=False)
        pd.DataFrame(rows_test_homo ).query("outer_fold == @ofold").to_csv(fold_dir/"homo/test_preds.csv", index=False)
        pd.DataFrame(rows_train_homo).query("outer_fold == @ofold").to_csv(fold_dir/"homo/train_preds.csv", index=False)

        print(f"[{tag}] Fold {ofold}: "
              f"HETERO R2={r2_he:.3f} NLL={nll_he:.3f} | "
              f"HOMO R2={r2_ho:.3f} NLL={nll_ho:.3f}")

    res_he = pd.DataFrame(results_hetero)
    res_ho = pd.DataFrame(results_homo)
    res_he.to_csv(model_dir/"hetero_results.csv", index=False)
    res_ho.to_csv(model_dir/"homo_results.csv", index=False)

    test_he  = pd.DataFrame(rows_test_hetero);  train_he = pd.DataFrame(rows_train_hetero)
    test_ho  = pd.DataFrame(rows_test_homo);    train_ho = pd.DataFrame(rows_train_homo)
    test_he.to_csv(model_dir/"hetero_predictions_test.csv", index=False)
    train_he.to_csv(model_dir/"hetero_predictions_train.csv", index=False)
    test_ho.to_csv(model_dir/"homo_predictions_test.csv", index=False)
    train_ho.to_csv(model_dir/"homo_predictions_train.csv", index=False)

    # Plots — HETERO
    title_he = (f"Heteroscedastic — {tag.replace('_',' ').upper()}  "
                f"R²={res_he['R2'].mean():.3f}±{res_he['R2'].std():.3f}, "
                f"NLL={res_he['NLL'].mean():.3f}±{res_he['NLL'].std():.3f}")
    he_dir = model_dir/"hetero_plots"; he_dir.mkdir(exist_ok=True)
    plot_pred_vs_obs_with_bars(test_he, title_he, he_dir)
    plot_reliability(test_he, title_he, he_dir)
    plot_fold_grid(train_he, test_he, title_he, he_dir)
    self_check_and_save(train_he, test_he, he_dir)

    # Plots — HOMO
    title_ho = (f"Homoscedastic — {tag.replace('_',' ').upper()}  "
                f"R²={res_ho['R2'].mean():.3f}±{res_ho['R2'].std():.3f}, "
                f"NLL={res_ho['NLL'].mean():.3f}±{res_ho['NLL'].std():.3f}")
    ho_dir = model_dir/"homo_plots"; ho_dir.mkdir(exist_ok=True)
    plot_pred_vs_obs_with_bars(test_ho, title_ho, ho_dir)
    plot_reliability(test_ho, title_ho, ho_dir)
    plot_fold_grid(train_ho, test_ho, title_ho, ho_dir)
    self_check_and_save(train_ho, test_ho, ho_dir)

    # Summary
    summ = {
        "total_seconds": time.time() - t0,
        "hetero": {
            "R2_mean": float(res_he["R2"].mean()),  "R2_std":  float(res_he["R2"].std()),
            "MSE_mean": float(res_he["MSE"].mean()),"MSE_std":  float(res_he["MSE"].std()),
            "NLL_mean": float(res_he["NLL"].mean()),"NLL_std":  float(res_he["NLL"].std())
        },
        "homo": {
            "R2_mean": float(res_ho["R2"].mean()),  "R2_std":  float(res_ho["R2"].std()),
            "MSE_mean": float(res_ho["MSE"].mean()),"MSE_std":  float(res_ho["MSE"].std()),
            "NLL_mean": float(res_ho["NLL"].mean()),"NLL_std":  float(res_ho["NLL"].std())
        }
    }
    with open(model_dir/"meta"/"summary.json","w") as f: json.dump(summ, f, indent=2)
    (model_dir/"meta"/"DONE").write_text("ok\n")

    return res_he, res_ho

# --------------------- main ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    args = parser.parse_args()

    if os.getenv("PLOT_ONLY","0") == "1":
        run_dir = Path(args.run_dir or os.getenv("RUN_DIR",""))
        assert run_dir.exists(), "PLOT_ONLY=1 needs --run_dir or RUN_DIR=<path>"
        print("PLOT_ONLY mode expects previously saved CSVs; plots are generated during normal runs.")
        return

    # fresh run
    assert X_PATH.exists() and Y_PATH.exists(), "Missing input CSVs"
    X_df = pd.read_csv(X_PATH, index_col=0)
    y_df = pd.read_csv(Y_PATH, index_col=0)
    common = X_df.index.intersection(y_df.index)
    X = X_df.loc[common].to_numpy(dtype=float)
    y = y_df.loc[common].iloc[:,0].to_numpy(dtype=float)

    run_dir = OUT_ROOT / f"{timestamp()}_gpr_nested_mean_fixed"
    run_dir.mkdir(parents=True, exist_ok=True)

    # simple tee logger
    class Tee:
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, "a", buffering=1)
        def write(self, m): self.terminal.write(m); self.log.write(m)
        def flush(self): self.terminal.flush(); self.log.flush()
        def close(self): self.log.close()
    tee = Tee(run_dir/"run.log"); sys.stdout = tee
    try:
        print("Starting GPR UQ run (NO PCA; mean & epistemic fixed across homo/hetero)")
        print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=43)

        all_summaries = []
        for tag in ["rbf_rq_white"]:
            print(f"\n=== Kernel: {tag} ===")
            res_he, res_ho = run_model(tag, X, y, outer_cv, inner_cv, run_dir)
            all_summaries.append({
                "kernel": tag,
                "hetero_R2_mean":  res_he["R2"].mean(),  "hetero_R2_std":  res_he["R2"].std(),
                "hetero_NLL_mean": res_he["NLL"].mean(), "hetero_NLL_std": res_he["NLL"].std(),
                "homo_R2_mean":    res_ho["R2"].mean(),  "homo_R2_std":    res_ho["R2"].std(),
                "homo_NLL_mean":   res_ho["NLL"].mean(), "homo_NLL_std":   res_ho["NLL"].std(),
            })

        pd.DataFrame(all_summaries).to_csv(run_dir/"summary_across_kernels.csv", index=False)
        print("\nDone. Artifacts in:", run_dir)
        print("Training completed successfully!")
    finally:
        sys.stdout = tee.terminal
        tee.close()

if __name__ == "__main__":
    main()
