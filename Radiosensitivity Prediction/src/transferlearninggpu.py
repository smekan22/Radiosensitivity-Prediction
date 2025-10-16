import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random
import os
import time

# Set reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# GPU settings for better performance
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"🎲 All random seeds set to: {RANDOM_SEED}")
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class SimpleRadiosensitivityDataset(Dataset):
    """Simple dataset for latent features -> AUC prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class GPUOptimizedTransferLearningPredictor(nn.Module):
    """
    GPU-optimized neural network for radiosensitivity prediction
    Fixed BatchNorm issues and optimized for GPU training
    """
    
    def __init__(self, input_dim=200, hidden_dims=[256, 128], dropout_rate=0.3):
        super(GPUOptimizedTransferLearningPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Use LayerNorm instead of BatchNorm to avoid batch size issues
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for better GPU performance
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 GPU-optimized model: {total_params:,} trainable parameters")
        print(f"   Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
        
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class GPURadiosensitivityTransferLearning:
    """GPU-optimized transfer learning approach"""
    
    def __init__(self, output_dir="results_transfer_learning_gpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🚀 GPU acceleration enabled!")
        
    def cross_validate_with_mosa_features(self, X_latent, y_auc, n_folds=5, 
                                        batch_size=64, epochs=200, lr=0.001, 
                                        hidden_dims=[256, 128], dropout_rate=0.2, 
                                        patience=15, weight_decay=1e-4):
        """GPU-optimized cross-validation"""
        
        print(f"\n🧬 GPU Transfer Learning with MOSA Latent Features")
        print(f"   Input: {X_latent.shape[0]} samples × {X_latent.shape[1]} latent features")
        print(f"   Strategy: GPU-accelerated neural network")
        print(f"   Cross-validation: {n_folds} folds")
        print(f"   Batch size: {batch_size} (optimized for GPU)")
        print("="*70)
        
        start_time = time.time()
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X_latent)):
            fold_start = time.time()
            print(f"\n📁 Fold {fold + 1}/{n_folds}")
            print("-" * 30)
            
            # Split data
            X_train_raw, X_test_raw = X_latent[train_idx], X_latent[test_idx]
            y_train, y_test = y_auc[train_idx], y_auc[test_idx]
            
            # Split training into train/val (80/20)
            val_split = int(0.2 * len(X_train_raw))
            X_val_raw, X_train_raw = X_train_raw[:val_split], X_train_raw[val_split:]
            y_val, y_train = y_train[:val_split], y_train[val_split:]
            
            # Scale data (no leakage)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            print(f"   Train: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
            
            # Create data loaders with GPU-optimized settings
            train_dataset = SimpleRadiosensitivityDataset(X_train_scaled, y_train)
            val_dataset = SimpleRadiosensitivityDataset(X_val_scaled, y_val)
            test_dataset = SimpleRadiosensitivityDataset(X_test_scaled, y_test)
            
            # Optimized for GPU
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=torch.cuda.is_available()
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2 if torch.cuda.is_available() else 0,
                pin_memory=torch.cuda.is_available()
            )
            
            # Create model
            model = GPUOptimizedTransferLearningPredictor(
                input_dim=X_latent.shape[1], 
                hidden_dims=hidden_dims, 
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Train model
            train_history = self._train_fold_gpu(
                model, train_loader, val_loader, epochs, lr, patience, weight_decay
            )
            
            # Evaluate on test set
            test_results = self._evaluate_model_gpu(model, test_loader)
            
            fold_time = time.time() - fold_start
            
            fold_result = {
                'fold': fold + 1,
                'train_history': train_history,
                'test_results': test_results,
                'fold_time': fold_time
            }
            fold_results.append(fold_result)
            
            print(f"   ✅ Fold {fold + 1} Results (⏱️ {fold_time:.1f}s):")
            print(f"      R² = {test_results['r2']:.4f}")
            print(f"      RMSE = {test_results['rmse']:.4f}")
            print(f"      MAE = {test_results['mae']:.4f}")
            print(f"      Epochs trained: {train_history['epochs_trained']}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Aggregate results
        return self._aggregate_results(fold_results)
    
    def _train_fold_gpu(self, model, train_loader, val_loader, epochs, lr, patience, weight_decay):
        """GPU-optimized training for one fold"""
        
        # GPU-optimized settings
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.7, verbose=False
        )
        
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        # Mixed precision for faster GPU training
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Use mixed precision if on GPU
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = model(batch_features)
                        loss = criterion(predictions, batch_targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(batch_features)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            predictions = model(batch_features)
                            loss = criterion(predictions, batch_targets)
                    else:
                        predictions = model(batch_features)
                        loss = criterion(predictions, batch_targets)
                    
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            val_r2 = r2_score(val_targets, val_predictions)
            val_r2_scores.append(val_r2)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 25 == 0:
                print(f"      Epoch {epoch:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, R²={val_r2:.4f}")
            
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_r2': max(val_r2_scores) if val_r2_scores else 0,
            'epochs_trained': epoch + 1
        }
    
    def _evaluate_model_gpu(self, model, test_loader):
        """GPU-optimized model evaluation"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        batch_predictions = model(batch_features)
                else:
                    batch_predictions = model(batch_features)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'targets': targets
        }
    
    def _aggregate_results(self, fold_results):
        """Aggregate cross-validation results"""
        
        test_r2_scores = [result['test_results']['r2'] for result in fold_results]
        test_rmse_scores = [result['test_results']['rmse'] for result in fold_results]
        test_mae_scores = [result['test_results']['mae'] for result in fold_results]
        total_time = sum([result['fold_time'] for result in fold_results])
        
        summary = {
            'mean_r2': np.mean(test_r2_scores),
            'std_r2': np.std(test_r2_scores),
            'mean_rmse': np.mean(test_rmse_scores),
            'std_rmse': np.std(test_rmse_scores),
            'mean_mae': np.mean(test_mae_scores),
            'std_mae': np.std(test_mae_scores),
            'total_time': total_time,
            'fold_results': fold_results
        }
        
        print(f"\n" + "="*70)
        print("🏆 GPU TRANSFER LEARNING RESULTS")
        print("="*70)
        print(f"R² Score:  {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
        print(f"RMSE:      {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
        print(f"MAE:       {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
        print(f"Runtime:   {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"\n🚀 GPU-accelerated neural network trained on MOSA latent features!")
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _save_results(self, summary):
        """Save results to files"""
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Fold': [r['fold'] for r in summary['fold_results']],
            'R2': [r['test_results']['r2'] for r in summary['fold_results']],
            'RMSE': [r['test_results']['rmse'] for r in summary['fold_results']],
            'MAE': [r['test_results']['mae'] for r in summary['fold_results']],
            'Epochs': [r['train_history']['epochs_trained'] for r in summary['fold_results']],
            'Time_s': [r['fold_time'] for r in summary['fold_results']]
        })
        
        # Save detailed results
        results_df.to_csv(self.output_dir / 'gpu_transfer_learning_detailed_results.csv', index=False)
        
        # Save summary
        summary_df = pd.DataFrame({
            'Metric': ['R2', 'RMSE', 'MAE', 'Total_Time_min'],
            'Mean': [summary['mean_r2'], summary['mean_rmse'], summary['mean_mae'], summary['total_time']/60],
            'Std': [summary['std_r2'], summary['std_rmse'], summary['std_mae'], 0]
        })
        summary_df.to_csv(self.output_dir / 'gpu_transfer_learning_summary.csv', index=False)
        
        print(f"📁 Results saved to: {self.output_dir}/")
        
        return results_df

# Data loading functions (same as before but with correct paths)
def load_data_file(file_path):
    """Load CSV file, handling both gzipped and regular files"""
    try:
        file_path = Path(file_path)
        if file_path.suffix == '.gz':
            return pd.read_csv(file_path, compression='gzip')
        else:
            return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def choose_best_name(row, target_ids):
    """Choose the best name from model_name or synonyms"""
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
            return syn_clean
    
    return model_name_clean

def load_and_align_data():
    """Load and align data using the same method as baseline ML"""
    
    print("🚀 Loading and aligning data for GPU transfer learning...")
    print("="*60)
    
    # Updated paths for Myriad
    data_dir = Path("data")
    model_list_path = data_dir / "model_list_20250423.csv"
    target_auc_path = data_dir / "cleveland_auc_only.csv"
    latent_joint_path = data_dir / "latent_joint.csv.gz"
    
    # Load files
    print("Loading data files...")
    auc_data = load_data_file(target_auc_path)
    if auc_data is None:
        return None, None, None
    print(f"✓ Loaded AUC data: {auc_data.shape}")
    
    latent_data = load_data_file(latent_joint_path)
    if latent_data is None:
        return None, None, None
    
    # Handle index properly
    if latent_data.index.name != latent_data.columns[0]:
        latent_data = latent_data.set_index(latent_data.columns[0])
    
    print(f"✓ Loaded latent features: {latent_data.shape}")
    
    model_list = load_data_file(model_list_path)
    if model_list is None:
        return None, None, None
    print(f"✓ Loaded model list: {model_list.shape}")
    
    # Rest of alignment code (same as before)
    print("\nAligning datasets...")
    
    auc_id_col = auc_data.columns[0]
    auc_value_col = auc_data.columns[1]
    
    target_ids = set(
        auc_data[auc_id_col]
        .astype(str)
        .str.replace('-', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.upper()
    )
    
    print(f"AUC data: {len(target_ids)} unique clean target IDs")
    
    latent_data = latent_data.copy()
    first_col_sample = latent_data.iloc[0, 0]
    if isinstance(first_col_sample, (int, float)):
        print("Detected numerical values in first column - using index as model_id")
        latent_data = latent_data.reset_index()
        latent_data.columns.values[0] = 'model_id'
    else:
        latent_data.columns.values[0] = 'model_id'
    
    print(f"Latent data: {latent_data.shape[0]} samples")
    
    model_list = model_list.copy()
    model_list['preferred_name'] = model_list.apply(
        lambda row: choose_best_name(row, target_ids), axis=1
    )
    
    id_to_name = model_list.set_index('model_id')['preferred_name']
    latent_data['preferred_name'] = latent_data['model_id'].map(id_to_name)
    latent_data = latent_data.dropna(subset=['preferred_name'])
    
    matched_mask = latent_data['preferred_name'].isin(target_ids)
    matched_latent = latent_data[matched_mask].copy()
    
    print(f"✅ {matched_mask.sum()} latent samples match target dataset IDs")
    
    if matched_mask.sum() < 20:
        print("❌ Not enough matches found for reliable analysis")
        return None, None, None
    
    auc_subset = auc_data[auc_data[auc_id_col].astype(str).str.replace('-', '').str.replace(' ', '').str.upper().isin(matched_latent['preferred_name'])]
    auc_clean_ids = auc_subset[auc_id_col].astype(str).str.replace('-', '').str.replace(' ', '').str.upper()
    
    aligned_samples = []
    
    for _, latent_row in matched_latent.iterrows():
        pref_name = latent_row['preferred_name']
        
        auc_match = auc_clean_ids == pref_name
        if auc_match.any():
            auc_value = auc_subset[auc_match][auc_value_col].iloc[0]
            
            feature_cols = [col for col in latent_data.columns if col not in ['model_id', 'preferred_name']]
            features = latent_row[feature_cols].values
            
            aligned_samples.append({
                'features': features,
                'auc': auc_value,
                'sample_id': latent_row['model_id']
            })
    
    if len(aligned_samples) < 20:
        print("❌ Final alignment produced insufficient samples")
        return None, None, None
    
    X = np.array([s['features'] for s in aligned_samples])
    y = np.array([s['auc'] for s in aligned_samples])
    sample_ids = [s['sample_id'] for s in aligned_samples]
    
    print(f"✅ Successfully aligned {len(aligned_samples)} samples")
    print(f"   Features shape: {X.shape}")
    print(f"   AUC range: {y.min():.3f} to {y.max():.3f}")
    
    return X, y, sample_ids

# MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 GPU Transfer Learning for Radiosensitivity Prediction")
    print("Using MOSA latent features + GPU-accelerated neural network")
    print("="*70)
    
    # Load and align data
    X, y, sample_ids = load_and_align_data()
    
    if X is None:
        print("❌ Failed to load/align data. Please check file paths and data.")
        exit(1)
    
    # Run GPU transfer learning
    print(f"\n🚀 Starting GPU Transfer Learning...")
    predictor = GPURadiosensitivityTransferLearning()
    
    # GPU-optimized parameters
    results = predictor.cross_validate_with_mosa_features(
        X, y, 
        n_folds=5,              # Full 5-fold CV
        epochs=200,             # More epochs (faster on GPU)
        patience=15,            # More patience
        lr=0.001,               # Learning rate
        hidden_dims=[256, 128], # Larger model for GPU
        batch_size=64,          # Larger batch for GPU efficiency
        dropout_rate=0.2,       # Lower dropout
        weight_decay=1e-4       # L2 regularization
    )
    
    print(f"\n🎯 GPU Transfer Learning Complete!")
    print(f"📊 Final Performance: R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"⏱️  Total Runtime: {results['total_time']/60:.1f} minutes")
    print(f"📁 Detailed results saved in: results_transfer_learning_gpu/")