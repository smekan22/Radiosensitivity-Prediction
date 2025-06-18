#!/usr/bin/env python3

import os, sys
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from functools import partial
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *

WD = "Radiosensitivity Prediction"
DATA_DIR = f"{WD}/src/data/Cleveland"
RUN_DIR = f"{WD}/src/runs"
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

selected_genes = list(set(pd.read_csv(f"{DATA_DIR}/hugo_genes.txt", sep="\t")["symbol"]))
print(f"🧬 Loaded {len(selected_genes)} genes for analysis")

feature_names_path = f"{DATA_DIR}/feature_names.txt"
if not os.path.exists(feature_names_path):
    meta_df = pd.read_csv(f"{DATA_DIR}/metabolite_view_aligned.csv", index_col=0)
    with open(feature_names_path, 'w') as f:
        for col in meta_df.columns:
            f.write(f"{col}\n")
    print(f"✅ Created {feature_names_path}")

met_names = pd.read_csv(feature_names_path, header=None)[0].tolist()
print(f"🧬 Loaded {len(met_names)} metabolites")

def aggregate_results(tag):
    run_dir = os.path.join(WD, "runs") 
    results_path = os.path.join(run_dir, tag, "results.csv")
    output_path = os.path.join(run_dir, tag, "aggregated_test_results.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path, index_col=0)
        
        test_df = df[df['index'] == "test"]
        test_df = test_df.select_dtypes(include='number')
        mean_scores = test_df.mean()
        std_scores = test_df.std()

        summary_df = pd.DataFrame({
            "mean": mean_scores,
            "std": std_scores
        })
        summary_df.index.name = 'metric'
        summary_df.to_csv(output_path)
        
        print(f" Aggregated results for {tag}:")
        r2_col = 'r2' if 'r2' in mean_scores else 'auc_r2' if 'auc_r2' in mean_scores else None
        if r2_col:
            print(f"   R² = {mean_scores[r2_col]:.4f} ± {std_scores[r2_col]:.4f}")
        
        return mean_scores.get('r2', mean_scores.get('auc_r2', None))
    else:
        print(f"results.csv not found at: {results_path}")
        return None

config["data_dir"] = DATA_DIR
config["run_dir"] = RUN_DIR
config["views"] = [("gexpr", "gene_view_aligned.csv", selected_genes, 0, lambda x: x, lambda x: x),
                   ("metabolic", "metabolite_view_aligned.csv", met_names, 0, lambda x: x, lambda x: x)]
config["view_alignment_method"] = "drop samples"
config["labels"] = [("radiation_targets_aligned.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5 
config["outer_kfolds"] = 5 
config["use_validation_on_test"] = False
config["val_metric"] = lambda x: r2_score(x["val_df"].ys, x["val_preds"])
config["results_processors"] = [lambda x: save_results(x, save_supervised_result, 
                                                      {"r2": r2_score,
                                                       "explained_variance": explained_variance_score,
                                                       "mse": mean_squared_error,
                                                       "mae": mean_absolute_error},
                                                      "individual")]
config["task"] = "regression"

print("=" * 80)
print("COMBINED GENE + METABOLIC DATA → RADIATION SENSITIVITY PREDICTION")
print("Models: KRR, SVR, Random Forest, AdaBoost")
print("5x5 nested cross-validation with grid search (matching run_proteomics.py)")
print("=" * 80)

print("\n Running Kernel Ridge Regression...")
alphas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100] 
gs_params = {"model_params": {f"degree_{d}_alpha_{a}": {"kernel": "poly", "degree": d, "alpha": a}
                              for d in [1, 2, 3] for a in alphas}}
config["model"] = KernelRidge
config["run_id"] = "krr_combined"
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
krr_r2 = aggregate_results("krr_combined")


print("\n Running Support Vector Regression...")
Cs = [1 / a for a in alphas]  
gs_params = {
    "model_params": {
        f"degree_{d}_C_{c:.4f}": {
            "kernel": "poly",
            "degree": d,
            "C": c
        }
        for d in [1, 2, 3] for c in Cs
    }
}
config["model"] = SVR
config["run_id"] = "svr_combined"
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
svr_r2 = aggregate_results("svr_combined")

print("\n Running Random Forest...")
gs_params = {
    "model_params": {
        f"bootstrap_{b}_depth_{d}_estimators_{n}": {
            "bootstrap": b,
            "max_depth": d,
            "n_estimators": n
        }
        for b in [True, False]
        for d in [10, 30, 50, 70, None]
        for n in [10, 50, 100, 200]
    }
}
config["model"] = RandomForestRegressor
config["run_id"] = "random_forest_combined"
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
rf_r2 = aggregate_results("random_forest_combined")

print("\n Running AdaBoost...")
gs_params = {"model_params": {f"lr_{l}_estimators_{n}":
                              {"learning_rate": l, "n_estimators": n}
                              for l in [0.01, 0.05, 0.1, 0.3, 1] for n in [50, 100]}}
config["model"] = AdaBoostRegressor
config["run_id"] = "adaboost_combined"
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
ada_r2 = aggregate_results("adaboost_combined")

print("\n" + "=" * 80)
print(" COMBINED ANALYSIS COMPLETE!")
print("=" * 80)

print("\n FINAL MODEL COMPARISON:")
model_scores = {
    "KRR": krr_r2,
    "SVR": svr_r2,
    "Random Forest": rf_r2,
    "AdaBoost": ada_r2
}

valid_scores = [(name, score) for name, score in model_scores.items() if score is not None]
valid_scores.sort(key=lambda x: x[1], reverse=True)

if valid_scores:
    print("\nModel Rankings (by R²):")
    for i, (name, score) in enumerate(valid_scores, 1):
        print(f"  {i}. {name:<15} R² = {score:.4f}")
    
    comparison_df = pd.DataFrame(valid_scores, columns=["Model", "Test_R2"])
    comparison_path = os.path.join(WD, "combined_model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n Model comparison saved to: {comparison_path}")
    
    best_name, best_score = valid_scores[0]
    print(f"\n BEST MODEL: {best_name}")
    print(f"   Test R² = {best_score:.4f} ({best_score:.1%})")
else:
    print("\n No models completed successfully")

print(f"\n📁 Detailed results saved in: {RUN_DIR}/")
print("\nFor each model, check:")
print("  - results.csv: Full cross-validation results")
print("  - aggregated_test_results.csv: Summary statistics (mean ± std)")
print("  - config.txt: Best hyperparameters selected")