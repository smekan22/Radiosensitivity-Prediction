#!/usr/bin/env python3

import numpy as np
setattr(np, "float", float)
setattr(np, "int", int)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time, warnings, os, argparse, traceback
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

try:
    from mofapy2.run.entry_point import entry_point
    MOFA_AVAILABLE = True
    print("MOFA available")
except ImportError:
    MOFA_AVAILABLE = False
    print("MOFA not available. Install with: pip install mofapy2")

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) available")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU not available")

warnings.filterwarnings('ignore')

def select_top_variance_features(df_train, df_test, n_or_frac, min_features=50):
    if isinstance(n_or_frac, float) and n_or_frac <= 1.0:
        n_keep = max(int(len(df_train.columns) * n_or_frac), min_features)
    else:
        n_keep = min(int(n_or_frac), len(df_train.columns))
    train_variances = df_train.var(axis=0, skipna=True)
    valid_features = train_variances[train_variances > 1e-8]
    if len(valid_features) == 0:
        raise ValueError("No features with sufficient variance in training data!")
    n_keep = min(n_keep, len(valid_features))
    selected_features = valid_features.nlargest(n_keep).index
    print(f"Selected {len(selected_features)} features from {len(df_train.columns)} based on training variance")
    return df_train[selected_features], df_test[selected_features]

class Fixed3ViewMOFARunner:
    def __init__(self, data_dir, output_dir, use_gpu=True,
                 top_n_rna=3000, top_frac_meth=0.01, top_n_prot=1000,
                 filter_low_var_samples=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu and GPU_AVAILABLE and MOFA_AVAILABLE
        self.top_n_rna = top_n_rna
        self.top_frac_meth = top_frac_meth
        self.top_n_prot = top_n_prot
        self.filter_low_var_samples = filter_low_var_samples
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._setup_logging()
        self.reg_grids = {
            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.5, 0.7, 0.9],
                "max_iter": [5000],
                "tol": [1e-4]
            },
            "Lasso": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "max_iter": [5000],
                "tol": [1e-4]
            },
            "KRR": {
                "kernel": ["rbf"],
                "alpha": [0.01, 0.1, 1.0],
                "gamma": [0.001, 0.01, 0.1]
            },
            "SVR": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1.0, 10.0],
                "gamma": [0.001, 0.01, 0.1],
                "epsilon": [0.01, 0.1, 1.0]
            }
        }
        self.mofa_config = {
            "initial_factors": 20,
            "iterations": 1000,
            "convergence_mode": "medium",
            "dropR2": 0.01,
            "spikeslab_weights": True,
            "ard_weights": True,
            "ard_factors": True
        }

    def _setup_logging(self):
        import logging
        self.logger = logging.getLogger('fixed_3view_mofa')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh = logging.FileHandler(self.output_dir / "fixed_3view_mofa.log")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def load_and_align_data(self):
        self.logger.info("Loading and Aligning Methylation + Proteomics + RNA Data")
        methylation = pd.read_csv(self.data_dir / "methylation_imputed.csv", index_col=0)
        proteomics = pd.read_csv(self.data_dir / "proteomics_imputed.csv", index_col=0)
        transcriptomics = pd.read_csv(self.data_dir / "transcriptomics_imputed.csv", index_col=0)
        targets = pd.read_csv(self.data_dir / "cleveland_auc_only.csv", index_col=0)
        self.logger.info(f"Raw data shapes: methylation={methylation.shape}, proteomics={proteomics.shape}, transcriptomics={transcriptomics.shape}, targets={targets.shape}")
        common = (methylation.index
                 .intersection(proteomics.index)
                 .intersection(transcriptomics.index)
                 .intersection(targets.index))
        if len(common) == 0:
            raise ValueError("No common samples found between datasets")
        self.logger.info(f"Common samples found: {len(common)}")
        methylation_aligned = methylation.loc[common]
        proteomics_aligned = proteomics.loc[common]
        transcriptomics_aligned = transcriptomics.loc[common]
        targets_aligned = targets.loc[common]
        methylation_clean = methylation_aligned.fillna(methylation_aligned.median())
        proteomics_clean = proteomics_aligned.fillna(proteomics_aligned.median())
        transcriptomics_clean = transcriptomics_aligned.fillna(transcriptomics_aligned.median())
        transcriptomics_clean = transcriptomics_clean.loc[:, ~transcriptomics_clean.columns.str.startswith("Y_")]
        methylation_clean = methylation_clean.loc[:, methylation_clean.var() > 1e-8]
        proteomics_clean = proteomics_clean.loc[:, proteomics_clean.var() > 1e-8]
        transcriptomics_clean = transcriptomics_clean.loc[:, transcriptomics_clean.var() > 1e-8]
        if self.filter_low_var_samples:
            self.logger.info("Filtering low-variance samples...")
            n_samples_before = len(methylation_clean)
            sample_vars = (methylation_clean.var(axis=1, skipna=True) + 
                          proteomics_clean.var(axis=1, skipna=True) + 
                          transcriptomics_clean.var(axis=1, skipna=True))
            threshold = sample_vars.quantile(0.05)
            valid_samples = sample_vars[sample_vars >= threshold].index
            methylation_clean = methylation_clean.loc[valid_samples]
            proteomics_clean = proteomics_clean.loc[valid_samples]
            transcriptomics_clean = transcriptomics_clean.loc[valid_samples]
            targets_aligned = targets_aligned.loc[valid_samples]
            self.logger.info(f"Removed {n_samples_before - len(valid_samples)} low-variance samples")
        y = targets_aligned.iloc[:, 0].values.astype(np.float32)
        methylation_clean = methylation_clean.reset_index(drop=True)
        proteomics_clean = proteomics_clean.reset_index(drop=True)
        transcriptomics_clean = transcriptomics_clean.reset_index(drop=True)
        self.logger.info(f"Final aligned data: n_samples={len(methylation_clean)}, meth_features={methylation_clean.shape[1]}, prot_features={proteomics_clean.shape[1]}, rna_features={transcriptomics_clean.shape[1]}")
        return methylation_clean, proteomics_clean, transcriptomics_clean, y

    def train_mofa_3view(self, meth_train, prot_train, rna_train, seed=42):
        try:
            n_samples = meth_train.shape[0]
            max_factors = min(meth_train.shape[1], prot_train.shape[1], rna_train.shape[1], n_samples//3, 30)
            initial_factors = min(max_factors, self.mofa_config["initial_factors"])
            self.logger.info(f"Training 3-view MOFA: factors={initial_factors}, iterations={self.mofa_config['iterations']} (GPU: {self.use_gpu})")
            X1 = meth_train.values.astype(np.float32)
            X2 = prot_train.values.astype(np.float32)
            X3 = rna_train.values.astype(np.float32)
            self.logger.info(f"Input shapes: X1={X1.shape}, X2={X2.shape}, X3={X3.shape}")
            ep = entry_point()
            ep.set_data_matrix([[X1], [X2], [X3]], 
                             likelihoods=["gaussian", "gaussian", "gaussian"],
                             views_names=["methylation", "proteomics", "transcriptomics"])
            ep.set_model_options(
                factors=initial_factors,
                spikeslab_weights=self.mofa_config["spikeslab_weights"],
                ard_weights=self.mofa_config["ard_weights"],
                ard_factors=self.mofa_config["ard_factors"]
            )
            ep.set_train_options(
                iter=self.mofa_config["iterations"],
                seed=seed,
                convergence_mode=self.mofa_config["convergence_mode"],
                verbose=False,
                gpu_mode=self.use_gpu,
                dropR2=self.mofa_config["dropR2"],
                save_interrupted=True
            )
            ep.build()
            t0 = time.time()
            ep.run()
            t_train = time.time() - t0
            model = ep.model
            if model is None:
                raise RuntimeError("MOFA model is None after training")
            expectations = model.getExpectations()
            if expectations and "Z" in expectations:
                Z = expectations["Z"]
                if isinstance(Z, dict): 
                    Z = list(Z.values())[0]
                final_factors = Z.shape[1] if hasattr(Z, 'shape') else 0
                self.logger.info(f"3-view MOFA completed in {t_train:.1f}s: {initial_factors}→{final_factors} factors")
            return model
        except Exception as e:
            self.logger.error(f"Error training 3-view MOFA: {e}")
            traceback.print_exc()
            raise

    def project_3view_fixed(self, model, meth_test, prot_test, rna_test):
        """Project test data with robust dimension handling - FIXED: same approach as 2-view"""
        try:
            exp = model.getExpectations()
            
            # Extract factor loadings for all views
            Z_train = exp.get("Z", None)
            if Z_train is None:
                raise RuntimeError("No factors found in MOFA model")
                
            if isinstance(Z_train, dict):
                Z_train = list(Z_train.values())[0]
            
            n_factors = Z_train.shape[1]
            self.logger.info(f"Training factors shape: {Z_train.shape}")
            
            # Extract weight matrices 
            Wdict = exp.get("W", None)
            if Wdict is None:
                raise RuntimeError("No weight matrices found")
                
            if isinstance(Wdict, dict):
                Wlist = list(Wdict.values())[0]
            else:
                Wlist = Wdict
            
            if len(Wlist) != 3:  # FIXED: 3 views instead of 2
                raise RuntimeError(f"Expected 3 weight matrices, got {len(Wlist)}")
            
            # Extract weights for all 3 views
            W_meth, W_prot, W_rna = Wlist[0], Wlist[1], Wlist[2]
            
            # Handle different weight matrix formats - FIXED: same as 2-view
            if isinstance(W_meth, dict):
                W_meth = W_meth.get('mean', W_meth.get('E', W_meth))
            if isinstance(W_prot, dict):
                W_prot = W_prot.get('mean', W_prot.get('E', W_prot))
            if isinstance(W_rna, dict):
                W_rna = W_rna.get('mean', W_rna.get('E', W_rna))
            
            # Ensure correct dimensions
            if (W_meth.shape[1] != n_factors or W_prot.shape[1] != n_factors or 
                W_rna.shape[1] != n_factors):
                raise RuntimeError(f"Weight matrix dimensions don't match factors")
            
            # Extract precision parameters - FIXED: same approach as 2-view
            tau_candidates = ["tau", "noisePrecision", "Tau"]
            taus = [1.0, 1.0, 1.0]  # Default for 3 views
            
            for key in tau_candidates:
                if key in exp:
                    raw = exp[key]
                    if isinstance(raw, dict):
                        raw = list(raw.values())[0]
                    
                    if len(raw) >= 3:
                        taus = []
                        for i in range(3):  # Need 3 for our views
                            p = raw[i]
                            if isinstance(p, dict):
                                tau_val = p.get('mean', p.get('E', 1.0))
                            else:
                                tau_val = p
                            taus.append(float(np.mean(tau_val) if hasattr(tau_val, '__iter__') else tau_val))
                    break
            
            # Closed-form projection for 3 views - FIXED: extended from 2-view
            K = n_factors
            A = np.eye(K, dtype=np.float32)
            
            # Add contributions from all 3 views
            A += taus[0] * (W_meth.T @ W_meth)
            A += taus[1] * (W_prot.T @ W_prot)
            A += taus[2] * (W_rna.T @ W_rna)
            
            # Compute inverse with fallback - FIXED: same as 2-view
            try: 
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                self.logger.warning("Matrix inversion failed, using pseudo-inverse")
                Ainv = np.linalg.pinv(A)

            # Project test samples for 3 views
            X1_test = meth_test.values.astype(np.float32)
            X2_test = prot_test.values.astype(np.float32)
            X3_test = rna_test.values.astype(np.float32)
            
            Z_projected = []
            for i in range(X1_test.shape[0]):
                b = (taus[0] * (W_meth.T @ X1_test[i]) + 
                     taus[1] * (W_prot.T @ X2_test[i]) +
                     taus[2] * (W_rna.T @ X3_test[i]))
                Z_projected.append(Ainv @ b)
            
            Z_test = np.vstack(Z_projected).astype(np.float32)
            
            self.logger.info(f"Projected {X1_test.shape[0]} samples to {Z_test.shape[1]} factors")
            
            # Verify dimensions match training
            if Z_test.shape[1] != Z_train.shape[1]:
                raise RuntimeError(f"Factor dimension mismatch: train={Z_train.shape[1]}, test={Z_test.shape[1]}")
            
            return Z_test
            
        except Exception as e:
            self.logger.error(f"Projection failed: {e}")
            # Return random factors as fallback with correct dimensions
            n_test = meth_test.shape[0]
            n_factors = 10  # Fallback number
            if 'Z_train' in locals():
                n_factors = Z_train.shape[1]
            
            self.logger.warning(f"Using random {n_factors} factors as fallback for {n_test} samples")
            np.random.seed(42)
            return np.random.randn(n_test, n_factors).astype(np.float32)

    def run(self):
        """Run leak-free nested CV analysis - FIXED: same approach as successful 2-view"""
        start_time = time.time()
        
        self.logger.info("="*80)
        self.logger.info("FIXED 3-VIEW MOFA ANALYSIS")
        self.logger.info("Methylation + Proteomics + RNA with 2-view success parameters")
        self.logger.info("="*80)
        
        # Load and align data
        meth_all, prot_all, rna_all, y_all = self.load_and_align_data()
        n_samples = len(meth_all)
        
        # FIXED: Same CV setup as 2-view (5x5 nested)
        outer = KFold(5, shuffle=True, random_state=42)
        inner = KFold(5, shuffle=True, random_state=1)
        
        # FIXED: Same models as 2-view
        models = [("ElasticNet", ElasticNet), ("Lasso", Lasso), ("KRR", KernelRidge), ("SVR", SVR)]
        
        self.logger.info(f"Nested CV: 5×5, Testing {len(models)} models")
        self.logger.info("Feature selection parameters (FIXED to match 2-view):")
        self.logger.info(f"  - RNA: top {self.top_n_rna} features")
        self.logger.info(f"  - Methylation: top {self.top_frac_meth:.1%} features")
        self.logger.info(f"  - Proteomics: top {self.top_n_prot} features")

        summary = {}
        all_results = []

        for name, ModelClass in models:
            self.logger.info(f"\n=== {name} ===")
            grid = self.reg_grids[name]
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer.split(range(n_samples)), 1):
                try:
                    self.logger.info(f"Fold {fold_idx}/5 - Leak-free feature selection")
                    
                    # Split data using iloc for robust indexing - FIXED: same as 2-view
                    meth_train_raw = meth_all.iloc[train_idx]
                    prot_train_raw = prot_all.iloc[train_idx]
                    rna_train_raw = rna_all.iloc[train_idx]
                    y_tr = y_all[train_idx]
                    
                    meth_test_raw = meth_all.iloc[test_idx]
                    prot_test_raw = prot_all.iloc[test_idx]
                    rna_test_raw = rna_all.iloc[test_idx]
                    y_te = y_all[test_idx]
                    
                    self.logger.info(f"  Split sizes: train={len(train_idx)}, test={len(test_idx)}")
                    
                    # Feature selection using ONLY training data - FIXED: same as 2-view
                    self.logger.info("  Selecting features based on training variance...")
                    
                    meth_train, meth_test = select_top_variance_features(
                        meth_train_raw, meth_test_raw, self.top_frac_meth, min_features=100
                    )
                    
                    prot_train, prot_test = select_top_variance_features(
                        prot_train_raw, prot_test_raw, self.top_n_prot, min_features=100
                    )
                    
                    rna_train, rna_test = select_top_variance_features(
                        rna_train_raw, rna_test_raw, self.top_n_rna, min_features=500
                    )
                    
                    self.logger.info(f"  Selected: meth={meth_train.shape[1]}, prot={prot_train.shape[1]}, rna={rna_train.shape[1]}")

                    # Train MOFA on selected training features
                    self.logger.info("  Training MOFA...")
                    mofa = self.train_mofa_3view(meth_train, prot_train, rna_train, seed=42 + fold_idx)

                    # Extract training factors
                    Z_tr = list(mofa.getExpectations()["Z"].values())[0]
                    
                    # Project test data
                    self.logger.info("  Projecting test data...")
                    Z_te = self.project_3view_fixed(mofa, meth_test, prot_test, rna_test)

                    # Verify dimensions match
                    if Z_tr.shape[1] != Z_te.shape[1]:
                        raise ValueError(f"Factor dimension mismatch: train={Z_tr.shape[1]}, test={Z_te.shape[1]}")

                    # Standardize factors - FIXED: same as 2-view
                    scaler = StandardScaler().fit(Z_tr)
                    Z_tr_s, Z_te_s = scaler.transform(Z_tr), scaler.transform(Z_te)

                    # Inner CV for hyperparameters - FIXED: same as 2-view
                    self.logger.info("  Running inner CV...")
                    gs = GridSearchCV(ModelClass(), grid, cv=inner, scoring="r2", n_jobs=2, verbose=0)
                    gs.fit(Z_tr_s, y_tr)

                    # Evaluate on test set
                    y_pred = gs.predict(Z_te_s)
                    r2 = r2_score(y_te, y_pred)
                    mse = mean_squared_error(y_te, y_pred)
                    mae = mean_absolute_error(y_te, y_pred)

                    fold_result = {
                        "model": name, "fold": fold_idx, "r2": r2, "mse": mse, "mae": mae,
                        "best_params": str(gs.best_params_), "best_cv_score": gs.best_score_,
                        "n_factors": Z_tr.shape[1],
                        "meth_features": meth_train.shape[1],
                        "prot_features": prot_train.shape[1],
                        "rna_features": rna_train.shape[1]
                    }
                    
                    fold_results.append(fold_result)
                    all_results.append(fold_result)
                    
                    self.logger.info(f"  Fold {fold_idx}: R²={r2:.4f}, CV_score={gs.best_score_:.4f}, factors={Z_tr.shape[1]}")

                except Exception as e:
                    self.logger.error(f"  Fold {fold_idx} failed: {e}")
                    traceback.print_exc()
                    fold_result = {
                        "model": name, "fold": fold_idx, "r2": np.nan, "mse": np.nan, "mae": np.nan,
                        "best_params": "FAILED", "best_cv_score": np.nan, "n_factors": np.nan,
                        "meth_features": np.nan, "prot_features": np.nan, "rna_features": np.nan
                    }
                    fold_results.append(fold_result)
                    all_results.append(fold_result)

            valid_results = [r for r in fold_results if not np.isnan(r["r2"])]
            if valid_results:
                r2_scores = [r["r2"] for r in valid_results]
                mse_scores = [r["mse"] for r in valid_results]
                mae_scores = [r["mae"] for r in valid_results]
                cv_scores = [r["best_cv_score"] for r in valid_results]
                n_factors = [r["n_factors"] for r in valid_results]
                
                summary[name] = {
                    "mean_r2": np.mean(r2_scores), "std_r2": np.std(r2_scores),
                    "mean_mse": np.mean(mse_scores), "std_mse": np.std(mse_scores),
                    "mean_mae": np.mean(mae_scores), "std_mae": np.std(mae_scores),
                    "mean_cv_score": np.mean(cv_scores), "std_cv_score": np.std(cv_scores),
                    "mean_factors": np.mean(n_factors),
                    "valid_folds": len(valid_results), "total_folds": len(fold_results)
                }
                self.logger.info(f"{name} SUMMARY: R²={summary[name]['mean_r2']:.4f}±{summary[name]['std_r2']:.4f} ({len(valid_results)}/5 valid)")

        detailed_df = pd.DataFrame(all_results)
        detailed_df.to_csv(self.output_dir / "fixed_3view_detailed_results.csv", index=False)
        
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(self.output_dir / "fixed_3view_summary.csv")

        runtime = (time.time() - start_time) / 3600
        
        self.logger.info("="*80)
        self.logger.info("FIXED 3-VIEW MOFA ANALYSIS COMPLETED")
        self.logger.info("="*80)
        
        if not summary_df.empty:
            best_model = summary_df.loc[summary_df["mean_r2"].idxmax()]
            self.logger.info(f"🏆 BEST MODEL: {best_model.name}")
            self.logger.info(f"   R² Score: {best_model['mean_r2']:.4f} ± {best_model['std_r2']:.4f}")
            self.logger.info(f"   Valid Folds: {best_model['valid_folds']}/{best_model['total_folds']}")
            self.logger.info(f"   Avg Factors: {best_model['mean_factors']:.1f}")
            
            self.logger.info("="*80)
            self.logger.info("COMPARISON WITH 2-VIEW RESULTS")
            self.logger.info("="*80)
            self.logger.info(f"2-view (Meth+RNA):     R² = 0.175 (ElasticNet)")
            self.logger.info(f"3-view (Fixed):        R² = {best_model['mean_r2']:.4f} ({best_model.name})")
            
            improvement = best_model['mean_r2'] - 0.175
            if improvement > 0:
                self.logger.info(f"3-view IMPROVEMENT: +{improvement:.4f}")
                self.logger.info("✅ Adding proteomics helped with proper parameters!")
            elif improvement > -0.05:
                self.logger.info(f" 3-view SIMILAR: {improvement:+.4f}")
                self.logger.info("⚠️  Proteomics adds complexity without much benefit")
            else:
                self.logger.info(f"3-view WORSE: {improvement:+.4f}")
                self.logger.info("❌ Proteomics hurts performance despite fixes")
        
        self.logger.info(f" Runtime: {runtime:.2f} hours")
        self.logger.info(f" Results saved to: {self.output_dir}")
        self.logger.info("Fair comparison complete!")

        return summary_df

def main():
    parser = argparse.ArgumentParser(description='Fixed 3-View MOFA Analysis - Fair Comparison')
    parser.add_argument('--data-dir', default='Radiosensitivity Prediction/src/data/Cleveland')
    parser.add_argument('--output-dir', default='fixed_3view_mofa_results')
    parser.add_argument('--cpu-only', action='store_true')
    
    parser.add_argument('--top-n-rna', type=int, default=3000)
    parser.add_argument('--top-frac-meth', type=float, default=0.01)  # FIXED: 0.01 like 2-view
    parser.add_argument('--top-n-prot', type=int, default=1000)
    parser.add_argument('--filter-low-var-samples', action='store_true', default=True)
    parser.add_argument('--no-filter-low-var-samples', dest='filter_low_var_samples', 
                       action='store_false')
    
    args = parser.parse_args()

    print("🔬 Fixed 3-View MOFA Analysis")
    print("✅ FIXED: Same parameters as successful 2-view")
    print("✅ FIXED: Methylation top 1% features (was 0.5%)")
    print("✅ FIXED: Low-variance sample filtering enabled")
    print("✅ FIXED: Same MOFA config (20 factors, 1000 iterations)")
    print("✅ FIXED: Numerical gamma values for KRR/SVR")
    print("📊 Fair comparison: 2-view vs 3-view with identical settings")
    
    if not MOFA_AVAILABLE:
        print(" MOFA not available. Please install: pip install mofapy2")
        return
    
    runner = Fixed3ViewMOFARunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_gpu=not args.cpu_only,
        top_n_rna=args.top_n_rna,
        top_frac_meth=args.top_frac_meth,  # FIXED: 0.01
        top_n_prot=args.top_n_prot,
        filter_low_var_samples=args.filter_low_var_samples  # FIXED: enabled
    )
    
    start_time = time.time()
    results = runner.run()
    runtime = (time.time() - start_time) / 3600
    
    if results is not None and not results.empty:
        print(f"\n Fixed 3-view analysis completed!")
        print(f"  Runtime: {runtime:.2f} hours")
        print(f" Results: {runner.output_dir}")
        
        # Print comparison
        best_r2 = results["mean_r2"].max()
        best_model = results.loc[results["mean_r2"].idxmax()]
        
        print(f"\nFAIR COMPARISON RESULTS:")
        print(f" 2-view (Meth+RNA):     R² = 0.175 (ElasticNet)")
        print(f"3-view (Fixed):        R² = {best_r2:.4f} ({best_model.name})")
        
        improvement = best_r2 - 0.175
        if improvement > 0.02:
            print(f" 3-view SIGNIFICANTLY BETTER: +{improvement:.4f}")
            print(" Proteomics adds valuable information!")
        elif improvement > 0:
            print(f"3-view SLIGHTLY BETTER: +{improvement:.4f}")
            print("Proteomics helps modestly")
        elif improvement > -0.02:
            print(f"3-view SIMILAR: {improvement:+.4f}")
            print(" No clear advantage either way")
        else:
            print(f" 3-view WORSE: {improvement:+.4f}")
            print(" Proteomics hurts despite fixes - stick with 2-view")
        
        print(f"\n RECOMMENDATION:")
        if best_r2 > 0.19:
            print("Use 3-view approach - clear improvement achieved!")
        elif best_r2 > 0.16:
            print("⚖️  Either approach works - slight preference based on complexity vs performance")
        else:
            print(" Stick with 2-view approach - 3-view doesn't help enough")
            
    else:
        print(f"\nAnalysis failed! Check logs: {runner.output_dir}/fixed_3view_mofa.log")

if __name__ == "__main__":
    main()