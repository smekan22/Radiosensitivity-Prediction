#!/usr/bin/env python3

#DIABLO RNA + METH

import pandas as pd
import numpy as np
import warnings
import traceback
import uuid
from pathlib import Path
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import random
import matplotlib.pyplot as plt

from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

os.environ['R_ENABLE_JIT'] = '3'
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

warnings.filterwarnings("ignore")

class OptimizedDIABLO2Omics:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pandas2ri.activate()
        
        try:
            self.mixOmics = importr("mixOmics")
            print("mixOmics loaded successfully")
        except Exception as e:
            print(f" Failed to load mixOmics: {e}")
            print("Please install mixOmics in R first:")
            print("  R")
            print("  install.packages('BiocManager')")
            print("  BiocManager::install('mixOmics')")
            raise
        
        r("set.seed(42)")
        
        self._setup_r_environment()

        self.diablo_grid = {
            "ncomp": [2, 3],
            "correlation_strength": [0.1, 0.8],
            "methylation_features": [40, 60],
            "transcriptomics_features": [50, 80],
        }
        
        self.reg_grid = {
            "KRR": {
                "kernel": ["rbf"],
                "degree": [2],
                "alpha":  [0.1, 1],  
            },
            "Lasso": {
                "alpha": [0.01, 0.1, 1.0]
            },
            "ElasticNet": {
                "alpha": [0.1, 1.0],
                "l1_ratio": [0.3, 0.7]
            },
            "SVR": {
                "kernel": ["rbf"],
                "C": [1.0, 10.0],
                "gamma": ["scale"],
                "epsilon": [0.1]
            }
        }

    def _setup_r_environment(self):
        """Set up R environment with GPU/parallel optimization"""
        r("""
        # Enable parallel processing if available
        library(parallel)
        options(mc.cores = detectCores())
        
        # Function to create a clean isolated environment
        create_isolated_env <- function(env_name) {
            if (exists(env_name, envir = .GlobalEnv)) {
                rm(list = env_name, envir = .GlobalEnv)
            }
            assign(env_name, new.env(), envir = .GlobalEnv)
            return(get(env_name, envir = .GlobalEnv))
        }
        
        # Function to clean up an environment
        cleanup_env <- function(env_name) {
            if (exists(env_name, envir = .GlobalEnv)) {
                rm(list = ls(envir = get(env_name, envir = .GlobalEnv)), 
                   envir = get(env_name, envir = .GlobalEnv))
                rm(list = env_name, envir = .GlobalEnv)
            }
            gc()  # Force garbage collection
        }
        
        # Function to safely execute code in isolated environment
        safe_eval <- function(code, env_name) {
            env <- get(env_name, envir = .GlobalEnv)
            eval(parse(text = code), envir = env)
        }
        """)

    def _generate_unique_env_name(self):
        return f"diablo2_env_{uuid.uuid4().hex[:8]}"

    def load_raw_data(self):
        print("=== Loading 2-omics raw data (Methylation + RNA) ===")
        
        # Load the 2 omics types
        methylation = pd.read_csv(self.data_dir / "methylation_imputed.csv", index_col=0)
        transcriptomics = pd.read_csv(self.data_dir / "transcriptomics_imputed.csv", index_col=0)
        targets = pd.read_csv(self.data_dir / "cleveland_auc_only.csv", index_col=0)
        
        print(f"Raw shapes:")
        print(f"  Methylation: {methylation.shape}")
        print(f"  Transcriptomics (RNA): {transcriptomics.shape}")
        print(f"  Targets: {targets.shape}")
        
        methylation.index = methylation.index.astype(str).str.strip()
        transcriptomics.index = transcriptomics.index.astype(str).str.strip()
        targets.index = targets.index.astype(str).str.strip()
        
        common_samples = sorted(list(
            set(methylation.index) & set(transcriptomics.index) & set(targets.index)
        ))
        
        print(f"Common samples across all datasets: {len(common_samples)}")
        
        if len(common_samples) < 10:
            raise ValueError("Too few common samples!")
        
        methylation_aligned = methylation.loc[common_samples].copy()
        transcriptomics_aligned = transcriptomics.loc[common_samples].copy()
        targets_aligned = targets.loc[common_samples].copy()
        
        print(f"Aligned shapes:")
        print(f"  Methylation: {methylation_aligned.shape}")
        print(f"  Transcriptomics (RNA): {transcriptomics_aligned.shape}")
        print(f"  Targets: {targets_aligned.shape}")
        
        indices = [methylation_aligned.index, transcriptomics_aligned.index, targets_aligned.index]
        for i in range(len(indices)-1):
            assert list(indices[i]) == list(indices[i+1]), f"Index mismatch between datasets {i} and {i+1}!"
        
        self.methylation_raw = methylation_aligned
        self.transcriptomics_raw = transcriptomics_aligned
        self.targets_raw = targets_aligned.iloc[:, 0]  
        
        return len(common_samples)

    def preprocess_fold_data(self, methylation_fold, transcriptomics_fold, targets_fold):
        methylation_processed = methylation_fold.fillna(methylation_fold.median())
        transcriptomics_processed = transcriptomics_fold.fillna(transcriptomics_fold.median())
        
        print(f"    After preprocessing (only imputation):")
        print(f"      Methylation: {methylation_processed.shape}")
        print(f"      Transcriptomics (RNA): {transcriptomics_processed.shape}")
        
        return methylation_processed, transcriptomics_processed, targets_fold

    def apply_preprocessing_to_test(self, methylation_test, transcriptomics_test, targets_test,
                                   train_methylation_cols, train_transcriptomics_cols,
                                   train_methylation_medians, train_transcriptomics_medians):
        methylation_processed = methylation_test.fillna(train_methylation_medians)
        transcriptomics_processed = transcriptomics_test.fillna(train_transcriptomics_medians)
        methylation_processed = methylation_processed[train_methylation_cols]
        transcriptomics_processed = transcriptomics_processed[train_transcriptomics_cols]
        return methylation_processed, transcriptomics_processed, targets_test

    def create_r_data_safe(self, methylation_df, transcriptomics_df, targets_series, env_name):
        methylation_array = methylation_df.values.astype("float64")
        transcriptomics_array = transcriptomics_df.values.astype("float64")
        targets_array = targets_series.values.astype("float64")
        n_samples = len(methylation_df)
        sample_names = [f"Sample_{i}" for i in range(n_samples)]
        methylation_r_df = pd.DataFrame(
            methylation_array,
            index=sample_names,
            columns=[f"Meth_{i}" for i in range(methylation_array.shape[1])]
        )
        transcriptomics_r_df = pd.DataFrame(
            transcriptomics_array,
            index=sample_names,
            columns=[f"RNA_{i}" for i in range(transcriptomics_array.shape[1])]
        )
        targets_r_df = pd.DataFrame(
            targets_array.reshape(-1, 1),
            index=sample_names,
            columns=["AUC"]
        )
        r.assign("current_env_name", env_name)
        r("current_env <- create_isolated_env(current_env_name)")
        r.assign("methylation_df", methylation_r_df)
        r.assign("transcriptomics_df", transcriptomics_r_df)
        r.assign("targets_df", targets_r_df)
        r(f"""
        safe_eval('
            methylation_matrix <- as.matrix(methylation_df)
            transcriptomics_matrix <- as.matrix(transcriptomics_df)
            targets_matrix <- as.matrix(targets_df)
            
            X <- list(
                methylation = methylation_matrix,
                transcriptomics = transcriptomics_matrix
            )
        ', "{env_name}")
        """)
        r("rm(methylation_df, transcriptomics_df, targets_df)")
        return True

    def fit_diablo_safe(self, params, env_name):
        """Fit DIABLO model for 2 omics in an ISOLATED environment"""
        try:
            r.assign("ncomp", params["ncomp"])
            r.assign("corr_strength", params["correlation_strength"])
            r.assign("meth_features", params["methylation_features"])
            r.assign("trans_features", params["transcriptomics_features"])
            r(f"""
            safe_eval('
                design <- matrix(c(
                    0, corr_strength,
                    corr_strength, 0
                ), nrow=2, ncol=2)
                rownames(design) <- c("methylation", "transcriptomics")
                colnames(design) <- c("methylation", "transcriptomics")
                
                keepX <- list(
                    methylation = rep(meth_features, ncomp),
                    transcriptomics = rep(trans_features, ncomp)
                )
                
                diablo_model <- block.spls(
                    X = X,
                    Y = targets_matrix,
                    ncomp = ncomp,
                    keepX = keepX,
                    design = design,
                    mode = "regression"
                )
            ', "{env_name}")
            """)
            r("rm(ncomp, corr_strength, meth_features, trans_features)")
            return True
        except Exception as e:
            print(f"        DIABLO fitting failed: {e}")
            return False

    def extract_components_safe(self, env_name):
        """Extract DIABLO components from 2 omics in ISOLATED environment"""
        try:
            r(f"""
            safe_eval('
                methylation_components <- diablo_model$variates$methylation
                transcriptomics_components <- diablo_model$variates$transcriptomics
                combined_components <- cbind(methylation_components, transcriptomics_components)
            ', "{env_name}")
            """)
            r(f"""temp_components <- get("combined_components", envir = get("{env_name}"))""")
            combined_components = np.array(r("temp_components"))
            r("rm(temp_components)")
            return combined_components
        except Exception as e:
            print(f"        Component extraction failed: {e}")
            return None

    def predict_components_safe(self, model_env_name, data_env_name):
        try:
            r(f"""
            trained_model <- get("diablo_model", envir = get("{model_env_name}"))
            prediction_X <- get("X", envir = get("{data_env_name}"))
            """)
            r(f"""
            predictions <- predict(trained_model, newdata = prediction_X)
            pred_methylation_comp <- predictions$variates$methylation
            pred_transcriptomics_comp <- predictions$variates$transcriptomics
            pred_combined_comp <- cbind(pred_methylation_comp, pred_transcriptomics_comp)
            """)
            pred_components = np.array(r("pred_combined_comp"))
            r("rm(trained_model, prediction_X, predictions, pred_methylation_comp, pred_transcriptomics_comp, pred_combined_comp)")
            return pred_components
        except Exception as e:
            print(f"        Component prediction failed: {e}")
            return None

    def cleanup_r_environment(self, env_name):
        """Clean up isolated R environment with garbage collection"""
        try:
            r.assign("env_to_clean", env_name)
            r("cleanup_env(env_to_clean)")
            r("rm(env_to_clean)")
        except Exception as e:
            print(f"        Environment cleanup failed: {e}")

    def inner_search(self, train_indices, model_name, cv):
        """Inner CV search with OPTIMIZED parameter space for 2 omics"""
        print(f"    Running OPTIMIZED 2-omics inner CV for {model_name}...")
        
        Regr = {
            "KRR": KernelRidge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "SVR": SVR
        }[model_name]
        
        best = {"score": -np.inf, "diablo": None, "regress": None}
        tested_combinations = 0
        successful_combinations = 0
        
        total_diablo_combos = len(list(ParameterGrid(self.diablo_grid)))
        total_reg_combos = len(list(ParameterGrid(self.reg_grid[model_name])))
        total_combos = total_diablo_combos * total_reg_combos
        
        print(f"      Testing {total_combos} combinations ({total_diablo_combos} DIABLO × {total_reg_combos} {model_name})")
        
        for dp in ParameterGrid(self.diablo_grid):
            for rp in ParameterGrid(self.reg_grid[model_name]):
                tested_combinations += 1
                scores = []
                
                for inner_train_idx, inner_val_idx in cv.split(train_indices):
                    train_env = self._generate_unique_env_name()
                    val_env = self._generate_unique_env_name()
                    
                    try:
                        actual_train_idx = train_indices[inner_train_idx]
                        actual_val_idx = train_indices[inner_val_idx]
                        
                        methylation_train = self.methylation_raw.iloc[actual_train_idx]
                        transcriptomics_train = self.transcriptomics_raw.iloc[actual_train_idx]
                        targets_train = self.targets_raw.iloc[actual_train_idx]
                        
                        methylation_val = self.methylation_raw.iloc[actual_val_idx]
                        transcriptomics_val = self.transcriptomics_raw.iloc[actual_val_idx]
                        targets_val = self.targets_raw.iloc[actual_val_idx]
                        
                        methylation_train_proc, transcriptomics_train_proc, targets_train_proc = \
                            self.preprocess_fold_data(methylation_train, transcriptomics_train, targets_train)
                        
                        if (dp["methylation_features"] > methylation_train_proc.shape[1] or 
                            dp["transcriptomics_features"] > transcriptomics_train_proc.shape[1]):
                            scores.append(-np.inf)
                            continue
                        
                        methylation_val_proc, transcriptomics_val_proc, targets_val_proc = \
                            self.apply_preprocessing_to_test(
                                methylation_val, transcriptomics_val, targets_val,
                                methylation_train_proc.columns, transcriptomics_train_proc.columns,
                                methylation_train_proc.median(), transcriptomics_train_proc.median()
                            )
                        
                        success = self.create_r_data_safe(
                            methylation_train_proc, transcriptomics_train_proc, targets_train_proc, train_env
                        )
                        if not success:
                            scores.append(-np.inf)
                            continue
                        
                        if not self.fit_diablo_safe(dp, train_env):
                            scores.append(-np.inf)
                            continue
                        
                        Z_tr = self.extract_components_safe(train_env)
                        if Z_tr is None:
                            scores.append(-np.inf)
                            continue
                        
                        success = self.create_r_data_safe(
                            methylation_val_proc, transcriptomics_val_proc, targets_val_proc, val_env
                        )
                        if not success:
                            scores.append(-np.inf)
                            continue
                        
                        Z_vl = self.predict_components_safe(train_env, val_env)
                        if Z_vl is None:
                            scores.append(-np.inf)
                            continue
                        
                        if Z_tr.shape[1] != Z_vl.shape[1]:
                            scores.append(-np.inf)
                            continue
                        
                        scaler = StandardScaler()
                        Z_tr_scaled = scaler.fit_transform(Z_tr)
                        Z_vl_scaled = scaler.transform(Z_vl)
                        
                        reg = Regr(**rp)
                        reg.fit(Z_tr_scaled, targets_train_proc.values)
                        y_pred = reg.predict(Z_vl_scaled)
                        score = r2_score(targets_val_proc.values, y_pred)
                        scores.append(score)
                        successful_combinations += 1
                        
                    except Exception:
                        scores.append(-np.inf)
                    
                    finally:
                        self.cleanup_r_environment(train_env)
                        self.cleanup_r_environment(val_env)
                
                mean_score = np.mean(scores)
                if mean_score > best["score"]:
                    best.update(score=mean_score, diablo=dp, regress=rp)
                
                if tested_combinations % 30 == 0:
                    print(f"      Progress: {tested_combinations}/{total_combos} ({100*tested_combinations/total_combos:.1f}%) - Success: {successful_combinations}/{tested_combinations}")
        
        print(f"    Best {model_name} inner CV score: {best['score']:.4f}")
        print(f"    Final success rate: {successful_combinations}/{tested_combinations}")
        print(f"    Best DIABLO params: {best['diablo']}")
        print(f"    Best {model_name} params: {best['regress']}")
        
        return best

    def _make_plots(self, df, summary, param_analysis):
        fig1 = self.output_dir / "figure_performance_summary.png"
        agg = df.groupby("model")["test_r2"].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(6,4), dpi=300)
        plt.bar(agg["model"], agg["mean"], yerr=agg["std"], capsize=4)
        plt.ylabel("Test $R^2$")
        plt.xlabel("Model")
        plt.title("Nested CV Test $R^2$ by Model")
        plt.tight_layout()
        plt.savefig(fig1)
        plt.close()

        fig2 = self.output_dir / "figure_r2_boxplot.png"
        data = [grp["test_r2"].values for _, grp in df.groupby("model")]
        labels = [name for name, _ in df.groupby("model")]
        plt.figure(figsize=(6,4), dpi=300)
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel("Test $R^2$")
        plt.xlabel("Model")
        plt.title("Distribution of Test $R^2$ Across Outer Folds")
        plt.tight_layout()
        plt.savefig(fig2)
        plt.close()

        fig3 = self.output_dir / "figure_inner_vs_outer.png"
        plt.figure(figsize=(5,5), dpi=300)
        for model, grp in df.groupby("model"):
            plt.scatter(grp["inner_cv_score"], grp["test_r2"], label=model, alpha=0.8)
        plt.xlabel("Mean Inner-CV Score ($R^2$)")
        plt.ylabel("Outer Test $R^2$")
        plt.title("Inner Validation vs Outer Test Performance")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig3)
        plt.close()

        for model in df["model"].unique():
            sub = param_analysis[param_analysis["model"] == model]
            if sub.empty:
                continue
            pivot = sub.pivot(index="ncomp", columns="correlation_strength", values="test_r2")
            fig4 = self.output_dir / f"figure_param_heatmap_{model}.png"
            plt.figure(figsize=(6,4), dpi=300)
            plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
            plt.xticks(ticks=range(pivot.shape[1]), labels=[str(x) for x in pivot.columns])
            plt.yticks(ticks=range(pivot.shape[0]), labels=[str(x) for x in pivot.index])
            plt.colorbar(label="Mean Test $R^2$")
            plt.xlabel("Correlation strength")
            plt.ylabel("ncomp")
            plt.title(f"Hyperparameter Heatmap ({model})")
            plt.tight_layout()
            plt.savefig(fig4)
            plt.close()

        fig5 = self.output_dir / "figure_keepX_hist.png"
        plt.figure(figsize=(6,4), dpi=300)
        bins = 8
        plt.hist(df["methylation_features"], bins=bins, alpha=0.6, label="Methylation")
        plt.hist(df["transcriptomics_features"], bins=bins, alpha=0.6, label="RNA")
        plt.xlabel("keepX (selected features per component)")
        plt.ylabel("Count")
        plt.title("Distribution of Selected keepX Across Folds")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig5)
        plt.close()

        fig6 = self.output_dir / "figure_components_count.png"
        plt.figure(figsize=(6,4), dpi=300)
        plt.hist(df["n_components"], bins=6)
        plt.xlabel("Total DIABLO components (concatenated)")
        plt.ylabel("Count")
        plt.title("Components Used Across Folds")
        plt.tight_layout()
        plt.savefig(fig6)
        plt.close()

        print("Saved figures:")
        for p in [fig1, fig2, fig3, *[self.output_dir / f"figure_param_heatmap_{m}.png" for m in df['model'].unique()], fig5, fig6]:
            if p.exists():
                print(f"   - {p.name}")

    def run(self):
        """Run the complete OPTIMIZED nested CV analysis for 2 omics"""
        print("="*80)
        print("OPTIMIZED 2-OMICS DIABLO ANALYSIS")
        print("Methylation + Transcriptomics (RNA)")
        print("="*80)

        random.seed(42)
        np.random.seed(42)
        
        n_samples = self.load_raw_data()
        
        print(f"\nOPTIMIZATION SETTINGS:")
        print(f"- DIABLO grid combinations: {len(list(ParameterGrid(self.diablo_grid)))}")
        for model, grid in self.reg_grid.items():
            print(f"- {model} grid combinations: {len(list(ParameterGrid(grid)))}")
        
        outer = KFold(2, shuffle=True, random_state=42)
        inner = KFold(2, shuffle=True, random_state=1)
        
        results = []
        sample_indices = np.arange(n_samples)
        
        total_outer_folds = 2
        total_models = len(self.reg_grid)
        
        print(f"\nCROSS-VALIDATION: {total_outer_folds} outer × {len(list(inner.split(sample_indices)))} inner")
        print(f"TOTAL MODELS: {total_models}")
        
        for fo, (train_idx, test_idx) in enumerate(outer.split(sample_indices), 1):
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {fo}/{total_outer_folds}")
            print(f"{'='*60}")
            print(f"Training: {len(train_idx)} samples, Testing: {len(test_idx)} samples")

            for mi, model_name in enumerate(self.reg_grid, 1):
                print(f"\n MODEL {mi}/{total_models}: {model_name}")
                
                final_train_env = self._generate_unique_env_name()
                final_test_env = self._generate_unique_env_name()
                
                try:
                    best = self.inner_search(train_idx, model_name, inner)
                    
                    if best["score"] == -np.inf or best["diablo"] is None:
                        print(f" {model_name} failed: no valid hyperparameters found")
                        continue
                    
                    print(f"  🔧 Training final {model_name} model with best parameters...")
                    
                    methylation_train = self.methylation_raw.iloc[train_idx]
                    transcriptomics_train = self.transcriptomics_raw.iloc[train_idx]
                    targets_train = self.targets_raw.iloc[train_idx]
                    
                    methylation_test = self.methylation_raw.iloc[test_idx]
                    transcriptomics_test = self.transcriptomics_raw.iloc[test_idx]
                    targets_test = self.targets_raw.iloc[test_idx]
                    
                    methylation_train_proc, transcriptomics_train_proc, targets_train_proc = \
                        self.preprocess_fold_data(methylation_train, transcriptomics_train, targets_train)
                    
                    methylation_test_proc, transcriptomics_test_proc, targets_test_proc = \
                        self.apply_preprocessing_to_test(
                            methylation_test, transcriptomics_test, targets_test,
                            methylation_train_proc.columns, transcriptomics_train_proc.columns,
                            methylation_train_proc.median(), transcriptomics_train_proc.median()
                        )
                    
                    success = self.create_r_data_safe(
                        methylation_train_proc, transcriptomics_train_proc, targets_train_proc, final_train_env
                    )
                    if not success:
                        print(f"  {model_name} failed: R data creation error")
                        continue
                    
                    if not self.fit_diablo_safe(best["diablo"], final_train_env):
                        print(f"  {model_name} failed: DIABLO fitting error")
                        continue
                    
                    Z_tr = self.extract_components_safe(final_train_env)
                    if Z_tr is None:
                        print(f"  {model_name} failed: component extraction error")
                        continue
                    
                    success = self.create_r_data_safe(
                        methylation_test_proc, transcriptomics_test_proc, targets_test_proc, final_test_env
                    )
                    if not success:
                        print(f"  {model_name} failed: test R data creation error")
                        continue
                    
                    Z_te = self.predict_components_safe(final_train_env, final_test_env)
                    if Z_te is None:
                        print(f"  {model_name} failed: test component prediction error")
                        continue
                    
                    scaler = StandardScaler()
                    Z_tr_scaled = scaler.fit_transform(Z_tr)
                    Z_te_scaled = scaler.transform(Z_te)
                    
                    Regressor = {
                        "KRR": KernelRidge,
                        "Lasso": Lasso,
                        "ElasticNet": ElasticNet,
                        "SVR": SVR
                    }[model_name]
                    
                    final_reg = Regressor(**best["regress"])
                    final_reg.fit(Z_tr_scaled, targets_train_proc.values)
                    
                    y_pred = final_reg.predict(Z_te_scaled)
                    
                    test_r2 = r2_score(targets_test_proc.values, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(targets_test_proc.values, y_pred))
                    test_mae = mean_absolute_error(targets_test_proc.values, y_pred)
                    
                    results.append({
                        "model": model_name,
                        "fold": fo,
                        "test_r2": test_r2,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                        "inner_cv_score": best["score"],
                        "best_diablo_params": str(best["diablo"]),
                        "best_regress_params": str(best["regress"]),
                        "n_components": Z_tr.shape[1],
                        "ncomp": best["diablo"]["ncomp"],
                        "correlation_strength": best["diablo"]["correlation_strength"],
                        "methylation_features": best["diablo"]["methylation_features"],
                        "transcriptomics_features": best["diablo"]["transcriptomics_features"]
                    })
                    
                    print(f"   {model_name} Fold {fo}: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
                    print(f"     Components: {Z_tr.shape[1]}, Inner CV: {best['score']:.4f}")
                    print(f"     Best ncomp={best['diablo']['ncomp']}, corr={best['diablo']['correlation_strength']}")
                    print(f"     Features: Meth={best['diablo']['methylation_features']}, RNA={best['diablo']['transcriptomics_features']}")
                    
                except Exception as e:
                    print(f"  {model_name} failed with error: {e}")
                    continue
                
                finally:
                    self.cleanup_r_environment(final_train_env)
                    self.cleanup_r_environment(final_test_env)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "2omics_meth_rna_detailed_results.csv", index=False)
            
            summary = df.groupby("model").agg({
                "test_r2": ["mean", "std", "min", "max"],
                "test_rmse": ["mean", "std"],
                "test_mae": ["mean", "std"],
                "inner_cv_score": ["mean", "std"],
                "n_components": ["mean"],
                "ncomp": ["mean"],
                "correlation_strength": ["mean"],
                "methylation_features": ["mean"],
                "transcriptomics_features": ["mean"]
            }).round(4)
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            summary = summary.reset_index()
            
            success_counts = df.groupby("model").size().reset_index(name='n_successful_folds')
            summary = summary.merge(success_counts, on='model')
            summary.to_csv(self.output_dir / "2omics_meth_rna_summary.csv", index=False)
            
            param_analysis = df.groupby(["model", "ncomp", "correlation_strength"]).agg({
                "test_r2": "mean"
            }).reset_index().sort_values("test_r2", ascending=False)
            param_analysis.to_csv(self.output_dir / "2omics_meth_rna_parameter_analysis.csv", index=False)
            
            print("\n" + "="*80)
            print("2-OMICS DIABLO RESULTS")
            print("Methylation + Transcriptomics (RNA)")
            print("="*80)
            print("\nPERFORMANCE SUMMARY:")
            print(summary[["model", "test_r2_mean", "test_r2_std", "n_successful_folds"]].to_string(index=False))
            
            print("\nBEST PARAMETER COMBINATIONS:")
            print(param_analysis.head(10).to_string(index=False))

            # Make and save figures
            self._make_plots(df, summary, param_analysis)
            
            print(f"\n Results & figures saved to: {self.output_dir}")
            return summary
            
        else:
            print("\n❌ No successful results!")
            return None

def main():
    import argparse
    import sys
    import time
    
    parser = argparse.ArgumentParser(description='2-Omics DIABLO Analysis (Methylation + RNA)')
    parser.add_argument("--data-dir", default="data/Cleveland",
                       help="Data directory path")
    parser.add_argument("--output-dir", default="diablo_2omics_meth_rna_results",
                       help="Output directory path")
    
    args = parser.parse_args()
    
    try:
        print("Starting 2-Omics DIABLO Analysis...")
        print("Omics types: Methylation + Transcriptomics (RNA)")
        start_time = time.time()
        
        analyzer = OptimizedDIABLO2Omics(args.data_dir, args.output_dir)
        results = analyzer.run()
        
        end_time = time.time()
        runtime_hours = (end_time - start_time) / 3600
        
        print(f"\nTOTAL RUNTIME: {runtime_hours:.2f} hours")
            
        if results is not None:
            print(" 2-Omics DIABLO analysis completed successfully!")
        else:
            print("Analysis failed!")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
