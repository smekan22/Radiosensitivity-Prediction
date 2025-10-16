#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------- Reproducibility -----------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
print(f"🎲 All random seeds set to: {RANDOM_SEED}")

# ----------------- Plot style (blues only) -----------------
plt.style.use('default')
BLUES = ['#0B3C5D', '#1F5A89', '#2E75B6', '#4A90D6', '#77AEE6', '#A7C6ED']  # darkest → lightest

class RadiosensitivityPredictor:
    """
    Pipeline for predicting radiosensitivity AUC using MOSA latent features
    - Nested CV with proper leakage control (scaler inside Pipeline)
    - Collects OOF predictions per model
    - Produces 2 publication-ready figures (blues only)
    """
    def __init__(self, data_dir="data", output_dir="results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Your paths
        self.model_list_path = self.data_dir / "clines/model_list_20230505.csv"
        self.target_auc_path = self.data_dir / "cleveland_auc_only.csv"
        self.latent_joint_path = "reports/sadafrun/files/20250811_111807_latent_joint.csv.gz"

        self.results = {}

    # ----------------- Loading / Alignment -----------------
    def load_data_file(self, file_path):
        try:
            file_path = Path(file_path)
            if file_path.suffix == '.gz':
                return pd.read_csv(file_path, compression='gzip')
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def choose_best_name(self, row, target_ids):
        model_id = row['model_id']
        model_name_clean = row['model_name'].replace('-', '').replace(' ', '').upper()
        if model_name_clean in target_ids:
            return model_name_clean
        synonyms = str(row.get('synonyms', '')).split(';')
        for syn in synonyms:
            if pd.isna(syn) or syn.strip() == '':
                continue
            syn_clean = syn.strip().replace('-', '').replace(' ', '').upper()
            if syn_clean in target_ids:
                print(f"🔁 Using synonym '{syn_clean}' for model_id {model_id}")
                return syn_clean
        return model_name_clean

    def load_and_align_data(self):
        print("Loading data files...")
        auc_data = self.load_data_file(self.target_auc_path)
        latent_data = self.load_data_file(self.latent_joint_path)
        model_list = self.load_data_file(self.model_list_path)
        if auc_data is None or latent_data is None or model_list is None:
            return None, None, None

        # Ensure latent index as first col if needed
        if latent_data.index.name != latent_data.columns[0]:
            latent_data = latent_data.set_index(latent_data.columns[0])

        # ---- Align (your proven method) ----
        auc_id_col = auc_data.columns[0]  # e.g., 'id'
        auc_value_col = auc_data.columns[1]  # e.g., 'auc'
        target_ids = set(
            auc_data[auc_id_col].astype(str)
            .str.replace('-', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.upper()
        )

        # Prepare latent
        latent_data = latent_data.reset_index()
        latent_data.columns.values[0] = 'model_id'
        first_col_sample = latent_data.iloc[0, 1]
        if isinstance(first_col_sample, (int, float)):
            pass  # already correct layout

        # Preferred names
        model_list = model_list.copy()
        model_list['preferred_name'] = model_list.apply(
            lambda row: self.choose_best_name(row, target_ids), axis=1
        )
        id_to_name = model_list.set_index('model_id')['preferred_name']
        latent_data['preferred_name'] = latent_data['model_id'].map(id_to_name)
        latent_data = latent_data.dropna(subset=['preferred_name'])
        matched_mask = latent_data['preferred_name'].isin(target_ids)
        matched_latent = latent_data[matched_mask].copy()
        print(f"✅ {matched_mask.sum()} latent samples match target dataset IDs")

        if matched_mask.sum() < 20:
            print("❌ Not enough matches for reliable analysis")
            return None, None, None

        auc_subset = auc_data.copy()
        auc_subset['_clean_id'] = (
            auc_subset[auc_id_col].astype(str)
            .str.replace('-', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.upper()
        )

        aligned_samples = []
        feature_cols = [c for c in matched_latent.columns if c not in ['model_id', 'preferred_name']]
        for _, latent_row in matched_latent.iterrows():
            pref = latent_row['preferred_name']
            hit = auc_subset['_clean_id'] == pref
            if hit.any():
                auc_val = auc_subset.loc[hit, auc_value_col].iloc[0]
                feats = latent_row[feature_cols].values
                aligned_samples.append({
                    'features': feats,
                    'auc': auc_val,
                    'sample_id': latent_row['model_id'],
                    'preferred_name': pref
                })

        if len(aligned_samples) < 20:
            print("❌ Final alignment produced insufficient samples")
            return None, None, None

        X = np.array([s['features'] for s in aligned_samples], dtype=float)
        y = np.array([s['auc'] for s in aligned_samples], dtype=float)
        sample_ids = [s['sample_id'] for s in aligned_samples]

        print(f"\n✅ Data alignment successful!")
        print(f"   Final: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"   AUC range: {y.min():.3f} to {y.max():.3f}")
        return X, y, sample_ids

    # ----------------- Models -----------------
    def setup_models(self):
        models = {
            'KernelRidge': {
                'model': KernelRidge(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0],
                    'kernel': ['linear', 'rbf', 'polynomial'],
                    'degree': [2, 3],      # used when kernel='polynomial'
                    'gamma': [0.001, 0.01, 0.1]  # used when kernel='rbf'
                }
            },
            'Lasso': {
                'model': Lasso(max_iter=2000, random_state=RANDOM_SEED),
                'params': {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]}
            },
            'ElasticNet': {
                'model': ElasticNet(max_iter=2000, random_state=RANDOM_SEED),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'kernel': ['rbf'],
                    'C': [0.1, 1.0, 10.0]
                }
            },
        }
        return models

    # ----------------- Nested CV with OOF predictions -----------------
    def nested_cross_validation(self, X, y, models, outer_cv=5, inner_cv=3, n_jobs=1):
        print(f"\nStarting nested {outer_cv}-fold CV with inner {inner_cv}-fold...")
        outer_split = KFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED)
        inner_split = KFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)

        results = {}

        for model_name, cfg in models.items():
            print(f"\n🔄 Testing {model_name}...")
            oof_true, oof_pred, oof_fold = [], [], []
            outer_scores, best_params_list = [], []

            for fold, (tr, te) in enumerate(outer_split.split(X, y), 1):
                X_tr, X_te = X[tr], X[te]
                y_tr, y_te = y[tr], y[te]

                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', cfg['model'])
                ])
                params = {f'model__{k}': v for k, v in cfg['params'].items()}

                gs = GridSearchCV(
                    pipe, params, cv=inner_split, scoring='r2',
                    n_jobs=n_jobs, error_score='raise'
                )

                try:
                    gs.fit(X_tr, y_tr)
                    y_hat = gs.predict(X_te)
                except Exception as e:
                    print(f"    ❌ Fold {fold} failed: {e}")
                    continue

                r2  = r2_score(y_te, y_hat)
                mse = mean_squared_error(y_te, y_hat)
                mae = mean_absolute_error(y_te, y_hat)
                outer_scores.append({'r2': r2, 'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)})
                best_params_list.append(gs.best_params_)
                oof_true.extend(y_te.tolist())
                oof_pred.extend(y_hat.tolist())
                oof_fold.extend([fold]*len(y_te))
                print(f"    Fold {fold}: R²={r2:.4f} | RMSE={np.sqrt(mse):.4f}")

            if outer_scores:
                oof_df = pd.DataFrame({'fold': oof_fold, 'y_true': oof_true, 'y_pred': oof_pred})
                oof_df.to_csv(self.output_dir / f'oof_predictions_{model_name}.csv', index=False)

                results[model_name] = {
                    'scores': outer_scores,
                    'mean_r2': np.mean([s['r2'] for s in outer_scores]),
                    'std_r2':  np.std([s['r2'] for s in outer_scores]),
                    'mean_rmse': np.mean([s['rmse'] for s in outer_scores]),
                    'std_rmse':  np.std([s['rmse'] for s in outer_scores]),
                    'mean_mae': np.mean([s['mae'] for s in outer_scores]),
                    'std_mae':  np.std([s['mae'] for s in outer_scores]),
                    'best_params': best_params_list,
                    'oof_true': np.array(oof_true),
                    'oof_pred': np.array(oof_pred)
                }
                print(f"  ✅ {model_name}: R² = {results[model_name]['mean_r2']:.4f} ± {results[model_name]['std_r2']:.4f}")
            else:
                print(f"  ❌ {model_name}: all folds failed")

        return results

    # ----------------- Summary + 2 Figures -----------------
    def create_results_summary(self, results):
        if not results:
            print("No results to summarize.")
            return None, None

        rows = []
        for m, r in results.items():
            rows.append({
                'Model': m,
                'Mean_R2': r['mean_r2'],
                'Std_R2': r['std_r2'],
                'Mean_RMSE': r['mean_rmse'],
                'Std_RMSE': r['std_rmse'],
                'Mean_MAE': r['mean_mae'],
                'Std_MAE': r['std_mae']
            })
        summary = pd.DataFrame(rows).sort_values('Mean_R2', ascending=False)
        summary.to_csv(self.output_dir / 'model_comparison_summary_no_leakage.csv', index=False)

        print("\nModel Performance (by R²):")
        for _, row in summary.iterrows():
            print(f"  {row['Model']:<12} R²={row['Mean_R2']:.3f} ± {row['Std_R2']:.3f} | RMSE={row['Mean_RMSE']:.3f}")

        return summary, summary.iloc[0]['Model']

    def _plot_polynomial_true_vs_pred(self, y_true, y_pred, title, savepath):
        # Scatter
        fig, ax = plt.subplots(figsize=(8.5, 7))
        ax.scatter(y_true, y_pred, s=46, alpha=0.75, edgecolors='white',
                   linewidth=0.6, color=BLUES[3], label='Predictions')

        # Perfect prediction line
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.6, color=BLUES[5],
                label='Perfect')

        # Polynomial trend (degree=2) in blues
        pf = PolynomialFeatures(degree=2)
        Xp = pf.fit_transform(y_true.reshape(-1, 1))
        lin = LinearRegression().fit(Xp, y_pred)
        xs = np.linspace(lo, hi, 200)
        ys = lin.predict(pf.transform(xs.reshape(-1, 1)))
        ax.plot(xs, ys, color=BLUES[0], linewidth=2.2, label='Poly (deg 2)')

        r2 = r2_score(y_true, y_pred)
        ax.set_title(f'{title}\nR² = {r2:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('True AUC', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted AUC', fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_model_bar_chart(self, summary_df, savepath):
        # Bar chart (R² ± std), blues only
        models = summary_df['Model'].tolist()
        means  = summary_df['Mean_R2'].values
        stds   = summary_df['Std_R2'].values

        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        colors = [BLUES[min(i, len(BLUES)-1)] for i in range(len(models))]
        bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors,
                      edgecolor=BLUES[0], linewidth=1.0, alpha=0.9)

        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (s if np.isfinite(s) else 0) + 0.006,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison (R² ± SD)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(0.05, float(means.max() + stds.max() + 0.05)))
        ax.grid(True, axis='y', alpha=0.25)
        plt.xticks(rotation=20, ha='right')
        fig.tight_layout()
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def build_two_publication_plots(self, results, summary_df, best_model_name):
        # 1) Polynomial regression (best model OOF)
        y_true = results[best_model_name]['oof_true']
        y_pred = results[best_model_name]['oof_pred']
        poly_path = self.output_dir / f'poly_true_vs_pred_{best_model_name}.png'
        self._plot_polynomial_true_vs_pred(
            y_true, y_pred, f'{best_model_name}: Predicted vs True (OOF)', poly_path
        )

        # 2) Bar chart across models
        bar_path = self.output_dir / 'model_r2_bar_comparison.png'
        self._plot_model_bar_chart(summary_df, bar_path)

        print("\n📊 Figures saved:")
        print(f" - {poly_path.name}")
        print(f" - {bar_path.name}")

    # ----------------- Orchestration -----------------
    def run_complete_pipeline(self):
        print("🚀 Starting Radiosensitivity Prediction Pipeline (no leakage)")
        X, y, sample_ids = self.load_and_align_data()
        if X is None:
            print("❌ Pipeline failed: alignment error")
            return None

        models = self.setup_models()
        print(f"\n✅ Models configured: {', '.join(models.keys())}")

        results = self.nested_cross_validation(X, y, models, outer_cv=5, inner_cv=3, n_jobs=1)
        if not results:
            print("❌ No results produced")
            return None

        summary_df, best_model = self.create_results_summary(results)
        if summary_df is not None:
            self.build_two_publication_plots(results, summary_df, best_model)

        return {
            'results': results,
            'summary': summary_df,
            'best_model': best_model,
            'data_info': {'n_samples': X.shape[0], 'n_features': X.shape[1],
                          'target_range': (float(y.min()), float(y.max()))}
        }

# ----------------- Main -----------------
if __name__ == "__main__":
    predictor = RadiosensitivityPredictor(
        data_dir="data",
        output_dir="results"
    )
    out = predictor.run_complete_pipeline()
    if out:
        print("\n✅ Done. CSVs + PNGs in:", predictor.output_dir.resolve())
