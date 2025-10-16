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

# Set reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print(f"🎲 All random seeds set to: {RANDOM_SEED}")

class SimpleRadiosensitivityDataset(Dataset):
    """Simple dataset for latent features -> AUC prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SimpleTransferLearningPredictor(nn.Module):
    """
    Simple neural network for radiosensitivity prediction using MOSA latent features
    """
    
    def __init__(self, input_dim=200, hidden_dims=[128, 64], dropout_rate=0.2):
        super(SimpleTransferLearningPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Transfer learning model: {total_params:,} trainable parameters")
        print(f"   Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

class RadiosensitivityTransferLearning:
    """Transfer learning approach using MOSA latent features"""
    
    def __init__(self, output_dir="results_transfer_learning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Use CPU (fine for your dataset size)
        self.device = torch.device('cpu')
        print(f"🖥️  Using device: {self.device}")
    
    def _create_loss_curves_plot(self, all_train_losses, all_val_losses, config_name):
        """Create training/validation loss curves"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate mean and std across folds
        max_epochs = max(len(losses) for losses in all_train_losses)
        
        # Pad shorter sequences
        padded_train = []
        padded_val = []
        
        for train_loss, val_loss in zip(all_train_losses, all_val_losses):
            if len(train_loss) < max_epochs:
                train_loss = train_loss + [train_loss[-1]] * (max_epochs - len(train_loss))
            if len(val_loss) < max_epochs:
                val_loss = val_loss + [val_loss[-1]] * (max_epochs - len(val_loss))
            padded_train.append(train_loss)
            padded_val.append(val_loss)
        
        epochs = range(max_epochs)
        mean_train = np.mean(padded_train, axis=0)
        std_train = np.std(padded_train, axis=0)
        mean_val = np.mean(padded_val, axis=0)  
        std_val = np.std(padded_val, axis=0)
        
        # Training loss
        ax1.plot(epochs, mean_train, color='red', linewidth=2)
        ax1.fill_between(epochs, mean_train - std_train, mean_train + std_train,
                        color='red', alpha=0.2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('MSE Loss')  # Fixed: was "Cross Entropy"
        ax1.set_title('Training Loss Curves')
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        ax2.plot(epochs, mean_val, color='blue', linewidth=2)
        ax2.fill_between(epochs, mean_val - std_val, mean_val + std_val,
                        color='blue', alpha=0.2)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MSE Loss')  # Fixed: was "Cross Entropy"
        ax2.set_title('Validation Loss Curves')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{config_name}_loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def cross_validate_with_mosa_features(self, X_latent, y_auc, n_folds=5, 
                                        batch_size=64, epochs=100, lr=0.001, 
                                        hidden_dims=[128, 64], dropout_rate=0.2, patience=10):
        """Cross-validate using MOSA latent features - NO DATA LEAKAGE"""
        
        print(f"\n🧬 Transfer Learning with MOSA Latent Features")
        print(f"   Input: {X_latent.shape[0]} samples × {X_latent.shape[1]} latent features")
        print(f"   Strategy: Neural network on top of MOSA representations")
        print(f"   Cross-validation: {n_folds} folds")
        print("="*70)
        
        print(f"✅ Data prepared (scaling will be done per fold)")
        print(f"   AUC range: {y_auc.min():.3f} to {y_auc.max():.3f}")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X_latent)):
            print(f"\n📁 Fold {fold + 1}/{n_folds}")
            print("-" * 30)
            
            # Split data WITHOUT scaling
            X_train_raw, X_test_raw = X_latent[train_idx], X_latent[test_idx]
            y_train, y_test = y_auc[train_idx], y_auc[test_idx]
            
            # Further split training into train/val (80/20)
            val_split = int(0.2 * len(X_train_raw))
            X_val_raw, X_train_raw = X_train_raw[:val_split], X_train_raw[val_split:]
            y_val, y_train = y_train[:val_split], y_train[val_split:]
            
            # FIT SCALER ON TRAINING DATA ONLY - NO DATA LEAKAGE!
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            print(f"   Train: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
            print(f"   ✅ Scaler fitted on training data only")
            
            # Create data loaders
            train_dataset = SimpleRadiosensitivityDataset(X_train_scaled, y_train)
            val_dataset = SimpleRadiosensitivityDataset(X_val_scaled, y_val)
            test_dataset = SimpleRadiosensitivityDataset(X_test_scaled, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Create model
            model = SimpleTransferLearningPredictor(
                input_dim=X_latent.shape[1], 
                hidden_dims=hidden_dims, 
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Train model
            train_history = self._train_fold(model, train_loader, val_loader, epochs, lr, patience)
            
            # Evaluate on test set
            test_results = self._evaluate_model(model, test_loader)
            
            fold_result = {
                'fold': fold + 1,
                'train_history': train_history,
                'test_results': test_results
            }
            fold_results.append(fold_result)
            
            print(f"   ✅ Fold {fold + 1} Results:")
            print(f"      R² = {test_results['r2']:.4f}")
            print(f"      RMSE = {test_results['rmse']:.4f}")
            print(f"      MAE = {test_results['mae']:.4f}")
        
        # Aggregate results
        return self._aggregate_results(fold_results)
    
    def _train_fold(self, model, train_loader, val_loader, epochs, lr, patience):
        """Train model for one fold"""
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
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
            
            if epoch % 20 == 0:
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
    
    def _evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
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
        all_train_losses = [result['train_history']['train_losses'] for result in fold_results]
        all_val_losses = [result['train_history']['val_losses'] for result in fold_results]
        
        summary = {
            'mean_r2': np.mean(test_r2_scores),
            'std_r2': np.std(test_r2_scores),
            'mean_rmse': np.mean(test_rmse_scores),
            'std_rmse': np.std(test_rmse_scores),
            'mean_mae': np.mean(test_mae_scores),
            'std_mae': np.std(test_mae_scores),
            'fold_results': fold_results
        }
        
        print(f"\n" + "="*70)
        print("🏆 TRANSFER LEARNING RESULTS")
        print("="*70)
        print(f"R² Score:  {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
        print(f"RMSE:      {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
        print(f"MAE:       {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f}")
        print(f"\n💡 This neural network learned to predict AUC from MOSA latent features!")
        
        # Save results
        self._create_loss_curves_plot(all_train_losses, all_val_losses, "config1_batchnorm_relu")
        self._save_results(summary)
        
        return summary
    
    def _save_results(self, summary):
        """Save results to files"""
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Fold': [r['fold'] for r in summary['fold_results']],
            'R2': [r['test_results']['r2'] for r in summary['fold_results']],
            'RMSE': [r['test_results']['rmse'] for r in summary['fold_results']],
            'MAE': [r['test_results']['mae'] for r in summary['fold_results']]
        })
        
        # Save detailed results
        results_df.to_csv(self.output_dir / 'transfer_learning_detailed_results.csv', index=False)
        
        # Save summary
        summary_df = pd.DataFrame({
            'Metric': ['R2', 'RMSE', 'MAE'],
            'Mean': [summary['mean_r2'], summary['mean_rmse'], summary['mean_mae']],
            'Std': [summary['std_r2'], summary['std_rmse'], summary['std_mae']]
        })
        summary_df.to_csv(self.output_dir / 'transfer_learning_summary.csv', index=False)
        
        print(f"📁 Results saved to: {self.output_dir}/")
        
        return results_df

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
            print(f"🔁 Using synonym '{syn_clean}' instead of model name '{model_name_clean}' for model_id {model_id}")
            return syn_clean
    
    return model_name_clean

def load_and_align_data():
    """Load and align data using the same method as your baseline ML"""
    
    print("🚀 Loading and aligning data for transfer learning...")
    print("="*60)
    
    # Define paths (adjust as needed)
    data_dir = Path("data")
    model_list_path = data_dir / "clines/model_list_20230505.csv"
    target_auc_path = data_dir / "cleveland_auc_only.csv"
    latent_joint_path = "reports/sadafrun/files/20250811_111807_latent_joint.csv.gz" 
    
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
    
    # Align data using your proven method
    print("\nAligning datasets...")
    
    # Prepare AUC data
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
    
    # Prepare latent data
    latent_data = latent_data.copy()
    first_col_sample = latent_data.iloc[0, 0]
    if isinstance(first_col_sample, (int, float)):
        print("Detected numerical values in first column - using index as model_id")
        latent_data = latent_data.reset_index()
        latent_data.columns.values[0] = 'model_id'
    else:
        latent_data.columns.values[0] = 'model_id'
    
    print(f"Latent data: {latent_data.shape[0]} samples")
    
    # Add preferred_name column to model_list
    model_list = model_list.copy()
    model_list['preferred_name'] = model_list.apply(
        lambda row: choose_best_name(row, target_ids), axis=1
    )
    
    # Map model_id in latent data to preferred names
    id_to_name = model_list.set_index('model_id')['preferred_name']
    
    original_model_ids = latent_data['model_id'].copy()
    latent_data['preferred_name'] = latent_data['model_id'].map(id_to_name)
    
    # Remove rows where mapping failed
    latent_data = latent_data.dropna(subset=['preferred_name'])
    
    # Check matches
    matched_mask = latent_data['preferred_name'].isin(target_ids)
    matched_latent = latent_data[matched_mask].copy()
    
    print(f"✅ {matched_mask.sum()} latent samples match target dataset IDs")
    
    if matched_mask.sum() < 20:
        print("❌ Not enough matches found for reliable analysis")
        return None, None, None
    
    # Create final aligned dataset
    auc_subset = auc_data[auc_data[auc_id_col].astype(str).str.replace('-', '').str.replace(' ', '').str.upper().isin(matched_latent['preferred_name'])]
    auc_clean_ids = auc_subset[auc_id_col].astype(str).str.replace('-', '').str.replace(' ', '').str.upper()
    
    aligned_samples = []
    
    for _, latent_row in matched_latent.iterrows():
        pref_name = latent_row['preferred_name']
        
        # Find corresponding AUC value
        auc_match = auc_clean_ids == pref_name
        if auc_match.any():
            auc_value = auc_subset[auc_match][auc_value_col].iloc[0]
            
            # Get latent features (exclude model_id and preferred_name columns)
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
    
    # Convert to arrays
    X = np.array([s['features'] for s in aligned_samples])
    y = np.array([s['auc'] for s in aligned_samples])
    sample_ids = [s['sample_id'] for s in aligned_samples]
    
    print(f"✅ Successfully aligned {len(aligned_samples)} samples")
    print(f"   Features shape: {X.shape}")
    print(f"   AUC range: {y.min():.3f} to {y.max():.3f}")
    
    return X, y, sample_ids

# MAIN EXECUTION
if __name__ == "__main__":
    print("🧬 Transfer Learning for Radiosensitivity Prediction")
    print("Using MOSA latent features + neural network prediction head")
    print("="*70)
    
    # Load and align data
    X, y, sample_ids = load_and_align_data()
    
    if X is None:
        print("❌ Failed to load/align data. Please check file paths and data.")
        exit(1)
    
    # Run transfer learning
    print(f"\n🚀 Starting Transfer Learning...")
    predictor = RadiosensitivityTransferLearning()
    
    # Quick test run (reduce parameters for faster execution)
    results = predictor.cross_validate_with_mosa_features(
        X, y, 
        n_folds=5,          # Full 5-fold CV
        epochs=50,          # Reduced epochs for faster run
        patience=10,        # Early stopping
        lr=0.001,
        hidden_dims=[128, 64],
        batch_size=64,
        dropout_rate=0.2
    )
    
    print(f"\n🎯 Transfer Learning Complete!")
    print(f"📊 Final Performance: R² = {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"📁 Detailed results saved in: results_transfer_learning/")