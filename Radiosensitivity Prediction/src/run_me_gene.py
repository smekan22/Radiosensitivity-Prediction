#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *

WD       = "Radiosensitivity Prediction"
DATA_DIR = f"{WD}/data/Cleveland" 
RUN_DIR  = f"{WD}/src/runs"
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

selected_genes = list(set(pd.read_csv(f"{DATA_DIR}/hugo_genes.txt", sep="\t")["symbol"]))

def aggregate_results(tag):
    results_path = os.path.join(RUN_DIR, tag, "results.csv")
    output_path  = os.path.join(RUN_DIR, tag, "aggregated_test_results.csv")

    if not os.path.exists(results_path):
        print(f"results.csv not found at: {results_path}")
        return None
    try:
        df = pd.read_csv(results_path, index_col=0)
        test_df = df[df['index'] == "test"].select_dtypes(include='number')
        mean_scores = test_df.mean()
        std_scores  = test_df.std()

        pd.DataFrame({"mean": mean_scores, "std": std_scores}).to_csv(output_path)
        print(f"   Aggregated results for {tag}:")
        print(f"   R²  = {mean_scores.get('r2', np.nan):.4f} ± {std_scores.get('r2', np.nan):.4f}")
        print(f"   MSE = {mean_scores.get('mse', np.nan):.4f} ± {std_scores.get('mse', np.nan):.4f}")
        print(f"   Saved to: {output_path}")
        return float(mean_scores.get('r2', np.nan))
    except Exception as e:
        print(f"Error processing results for {tag}: {e}")
        return None

config["data_dir"] = DATA_DIR
config["run_dir"]  = RUN_DIR

# gene-only view; NO methylation
config["views"] = [
    ("gexpr", "cleveland_gene_expression.csv", selected_genes, 0, lambda x: x, lambda x: x)
]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_only.csv", 0)]

# CV setup
config["tv_split_seed"] = 42
config["inner_kfolds"]  = 5
config["outer_kfolds"]  = 5
config["use_validation_on_test"] = False
# make sure we don't keep any stray holdout param around
if "test_samples" in config: del config["test_samples"]

# metric + results logging
config["val_metric"] = lambda x: r2_score(x["val_df"].ys, x["val_preds"])
config["results_processors"] = [
    lambda x: save_results(
        x, save_supervised_result,
        {"r2": r2_score, "explained_variance": explained_variance_score,
         "mse": mean_squared_error, "mae": mean_absolute_error},
        "individual"
    )
]
config["task"] = "regression"

print("=" * 80)
print("GENE EXPRESSION -> RADIATION SENSITIVITY (AUC)")
print("Baseline models: KRR (poly), SVR (poly), Lasso, Elastic Net")
print("Protocol: 5×5 nested cross-validation | No standardization")
print("=" * 80)

print("\n Running Kernel Ridge Regression (poly kernel)...")
config["run_id"] = "krr"
config["model"]  = KernelRidge
krr_gs = {
    "model_params": {
        f"degree_{d}_alpha_{a}": {"kernel": "poly", "degree": d, "alpha": a}
        for d in [1, 2, 3]
        for a in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    }
}
config["grid_search"] = construct_gs_params(krr_gs)
try:
    MLPipeline(config).run_crossvalidation()
    krr_r2 = aggregate_results("krr")
except Exception as e:
    print(f" KRR failed: {e}")
    krr_r2 = None

print("\n Running Support Vector Regression (poly kernel)...")
config["run_id"] = "svr"
config["model"]  = SVR
alphas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
Cs = [1.0 / a for a in alphas]
svr_gs = {
    "model_params": {
        f"degree_{d}_C_{c:.4f}": {"kernel": "poly", "degree": d, "C": c}
        for d in [1, 2, 3] for c in Cs
    }
}
config["grid_search"] = construct_gs_params(svr_gs)
try:
    MLPipeline(config).run_crossvalidation()
    svr_r2 = aggregate_results("svr")
except Exception as e:
    print(f" SVR failed: {e}")
    svr_r2 = None

print("\n Running Lasso...")
config["run_id"] = "lasso"
config["model"]  = Lasso
lasso_alphas = np.logspace(-4, 2, 13)  # 1e-4 ... 1e2
lasso_gs = {
    "model_params": {
        f"alpha_{a:g}": {"alpha": float(a), "max_iter": 10000}
        for a in lasso_alphas
    }
}
config["grid_search"] = construct_gs_params(lasso_gs)
try:
    MLPipeline(config).run_crossvalidation()
    lasso_r2 = aggregate_results("lasso")
except Exception as e:
    print(f" Lasso failed: {e}")
    lasso_r2 = None

print("\n Running Elastic Net...")
config["run_id"] = "elasticnet"
config["model"]  = ElasticNet
en_alphas    = np.logspace(-4, 2, 10)
en_l1_ratios = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
en_gs = {
    "model_params": {
        f"alpha_{a:g}_l1_{l1:g}": {"alpha": float(a), "l1_ratio": float(l1), "max_iter": 10000}
        for a in en_alphas for l1 in en_l1_ratios
    }
}
config["grid_search"] = construct_gs_params(en_gs)
try:
    MLPipeline(config).run_crossvalidation()
    en_r2 = aggregate_results("elasticnet")
except Exception as e:
    print(f" Elastic Net failed: {e}")
    en_r2 = None

print("\n" + "=" * 80)
print(" GENE-ONLY BASELINE ANALYSIS COMPLETE!")
print("=" * 80)

print("\n FINAL MODEL COMPARISON (outer test R² mean):")
model_scores = {
    "KRR(poly)": krr_r2,
    "SVR(poly)": svr_r2,
    "Lasso":     lasso_r2,
    "ElasticNet": en_r2
}
valid = [(k, v) for k, v in model_scores.items() if v is not None and np.isfinite(v)]
valid.sort(key=lambda x: x[1], reverse=True)
if valid:
    for i, (name, score) in enumerate(valid, 1):
        print(f"  {i}. {name:<12} R² = {score:.4f}")
    pd.DataFrame(valid, columns=["Model", "Test_R2"]).to_csv(
        os.path.join(WD, "gene_only_model_comparison.csv"), index=False
    )
    print(f"\n Model comparison saved to: {os.path.join(WD, 'gene_only_model_comparison.csv')}")
else:
    print(" No models completed successfully")

print(f"\n Detailed results saved in: {RUN_DIR}/")
print("  - <model>/results.csv: Full nested CV results (all outer folds)")
print("  - <model>/aggregated_test_results.csv: Mean ± std over outer test folds")
print("  - <model>/<fold>/<cv>/config.txt: Selected hyperparameters per split")
