#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

def load_metabolic_data(excel_path):
    print("Loading metabolic data...")
    
    metabolic_df = pd.read_excel(excel_path, sheet_name='1-clean data', index_col=0)
    
    print(f"Loaded metabolic data: {metabolic_df.shape[0]} cell lines, {metabolic_df.shape[1]} metabolites")
    print(f"Sample cell lines: {list(metabolic_df.index[:5])}")
    print(f"Sample metabolites: {list(metabolic_df.columns[:5])}")
    
    return metabolic_df

def load_radiation_data(csv_path):
    """Load radiation sensitivity data"""
    print("Loading radiation sensitivity data...")
    
    radiation_df = pd.read_csv(csv_path, index_col='id')
    
    print(f"Loaded radiation data: {radiation_df.shape[0]} cell lines")
    print(f"Sample cell lines: {list(radiation_df.index[:5])}")
    
    return radiation_df

def match_cell_lines(metabolic_df, radiation_df):
    """Match cell lines between metabolic and radiation datasets"""
    print("Matching cell lines...")
    metabolic_lines = set(metabolic_df.index)
    radiation_lines = set(radiation_df.index)
    
    exact_matches = metabolic_lines.intersection(radiation_lines)
    print(f"Exact matches: {len(exact_matches)}")
    
    partial_matches = []
    for rad_line in radiation_lines:
        if rad_line not in exact_matches:
            matching_met_lines = [met_line for met_line in metabolic_lines 
                                if met_line.startswith(rad_line + '_')]
            if matching_met_lines:
                partial_matches.append((rad_line, matching_met_lines[0]))
    
    print(f"Partial matches: {len(partial_matches)}")
    
    cell_line_mapping = {}
    
    for line in exact_matches:
        cell_line_mapping[line] = line
    
    for rad_line, met_line in partial_matches:
        cell_line_mapping[rad_line] = met_line
    
    total_matches = len(cell_line_mapping)
    print(f"Total matched cell lines: {total_matches}")
    
    return cell_line_mapping

def create_matched_datasets(metabolic_df, radiation_df, cell_line_mapping):
    """Create matched datasets with same cell lines"""
    print("Creating matched datasets...")
    
    common_lines = list(cell_line_mapping.keys())
    
    metabolic_matched = metabolic_df.loc[[cell_line_mapping[line] for line in common_lines]]
    metabolic_matched.index = common_lines  # Rename to radiation IDs for consistency
    
    # Create matched radiation data
    radiation_matched = radiation_df.loc[common_lines]
    
    print(f"Matched dataset shape:")
    print(f"  Metabolic features: {metabolic_matched.shape}")
    print(f"  Radiation targets: {radiation_matched.shape}")
    
    return metabolic_matched, radiation_matched

def save_pnet_format(metabolic_df, radiation_df, output_dir):
    """Save data in PNET expected format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving data to {output_path}")
    
    # Save feature matrix (cell_lines x metabolites)
    features_path = output_path / 'metabolic_features.csv'
    metabolic_df.to_csv(features_path)
    print(f"Saved features: {features_path}")
    
    # Save target values (cell_lines x 1)
    targets_path = output_path / 'radiation_targets.csv'
    radiation_df.to_csv(targets_path)
    print(f"Saved targets: {targets_path}")
    
    # Save feature names
    feature_names_path = output_path / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for feature in metabolic_df.columns:
            f.write(f"{feature}\n")
    print(f"Saved feature names: {feature_names_path}")
    
    # Save cell line names
    cell_lines_path = output_path / 'cell_lines.txt'
    with open(cell_lines_path, 'w') as f:
        for cell_line in metabolic_df.index:
            f.write(f"{cell_line}\n")
    print(f"Saved cell lines: {cell_lines_path}")
    
    summary_path = output_path / 'data_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Metabolic Data for PNET\n")
        f.write("=======================\n\n")
        f.write(f"Number of cell lines: {metabolic_df.shape[0]}\n")
        f.write(f"Number of metabolites: {metabolic_df.shape[1]}\n")
        f.write(f"Target variable: Radiation sensitivity (AUC)\n\n")
        f.write("Files created:\n")
        f.write("- metabolic_features.csv: Feature matrix\n")
        f.write("- radiation_targets.csv: Target values\n")
        f.write("- feature_names.txt: List of metabolite names\n")
        f.write("- cell_lines.txt: List of cell line names\n\n")
        f.write("Data statistics:\n")
        f.write(f"- Feature range: [{metabolic_df.min().min():.3f}, {metabolic_df.max().max():.3f}]\n")
        f.write(f"- Target range: [{radiation_df.min().min():.3f}, {radiation_df.max().max():.3f}]\n")
        f.write(f"- Missing values in features: {metabolic_df.isnull().sum().sum()}\n")
        f.write(f"- Missing values in targets: {radiation_df.isnull().sum().sum()}\n")
    
    print(f"Saved summary: {summary_path}")
    
    return {
        'features': features_path,
        'targets': targets_path,
        'feature_names': feature_names_path,
        'cell_lines': cell_lines_path,
        'summary': summary_path
    }

def main():
    parser = argparse.ArgumentParser(description='Prepare metabolic data for PNET')
    parser.add_argument('--metabolic_file', 
                       default='Radiosensitivity Prediction/src/data/41591_2019_404_MOESM2_ESM.xlsx',
                       help='Path to metabolic Excel file')
    parser.add_argument('--radiation_file', 
                       default='Radiosensitivity Prediction/src/data/cleveland_auc_only.csv',
                       help='Path to radiation CSV file')
    parser.add_argument('--output_dir', 
                       default='Radiosensitivity Prediction/src/data/Cleveland',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.metabolic_file):
            print(f"ERROR: Metabolic file not found: {args.metabolic_file}")
            return
        if not os.path.exists(args.radiation_file):
            print(f"ERROR: Radiation file not found: {args.radiation_file}")
            return
            
        metabolic_df = load_metabolic_data(args.metabolic_file)
        radiation_df = load_radiation_data(args.radiation_file)
        
        cell_line_mapping = match_cell_lines(metabolic_df, radiation_df)
        
        if len(cell_line_mapping) == 0:
            print("ERROR: No matching cell lines found!")
            return
        
        metabolic_matched, radiation_matched = create_matched_datasets(
            metabolic_df, radiation_df, cell_line_mapping
        )
        
        output_files = save_pnet_format(metabolic_matched, radiation_matched, args.output_dir)
        
        print("\nData preparation completed successfully!")
        print(f"Processed {metabolic_matched.shape[0]} cell lines with {metabolic_matched.shape[1]} metabolites")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()