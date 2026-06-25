"""
Quick plotting script to visualize saved CV results without rerunning experiments
Loads from ./results_logs/ directory
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import plotting functions from algo2
from algo2 import (
    plot_complexity_bias_variance, 
    plot_learning_curve_3d,
    plot_hyperparameter_surface,
    list_available_logs,
    load_cv_results,
    load_feature_importance
)

def find_latest_file(pattern, log_dir='./results_logs'):
    """Find the most recent file matching pattern"""
    if not os.path.exists(log_dir):
        print(f"No logs directory found at {log_dir}")
        return None
    
    files = [f for f in os.listdir(log_dir) if f.startswith(pattern)]
    if not files:
        return None
    
    files.sort(reverse=True)
    return os.path.join(log_dir, files[0])

def interactive_menu():
    """Interactive menu to select plotting options"""
    print("\n" + "="*70)
    print("TAXI DURATION PREDICTION - RESULTS PLOTTER")
    print("="*70)
    
    # List available logs
    print("\nAvailable saved results:")
    logs = list_available_logs()
    
    if not any(logs.values()):
        print("No log files found. Run algo2.py first to generate results.")
        return
    
    print("\nOptions:")
    print("1. Plot complexity vs bias-variance (from latest run)")
    print("2. Plot learning curves (from latest run)")
    print("3. Plot hyperparameter surface (from latest run)")
    print("4. View model comparison results")
    print("5. View feature importance")
    print("6. Exit")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == '1':
        file = find_latest_file('complexity_analysis_top3')
        if file:
            print(f"\nLoading: {file}")
            try:
                df = pd.read_csv(file)
                # Plot each model separately for clarity
                models = df['model'].unique()
                for model_name in models:
                    model_data = df[df['model'] == model_name]
                    complexities = model_data['complexity'].values
                    train_errors = model_data['train_error'].values
                    val_errors = model_data['val_error'].values
                    
                    print(f"\nPlotting {model_name}...")
                    plot_complexity_bias_variance(complexities, train_errors, val_errors, 
                                                 save_path=f'./complexity_{model_name.replace(" ", "_")}.png')
            except KeyError as e:
                print(f"Error: Expected columns not found in CSV. Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error loading file: {e}")
    
    elif choice == '2':
        file = find_latest_file('learning_curve')
        if file:
            print(f"\nLoading: {file}")
            try:
                plot_learning_curve_3d(log_file=file)
            except Exception as e:
                print(f"Error: {e}")
    
    elif choice == '3':
        file = find_latest_file('hyperparam')
        if file:
            print(f"\nLoading: {file}")
            print("Note: Hyperparameter surface requires grid data. Create custom plot with arrays.")
        else:
            print("No hyperparameter grid data saved. This requires manual data preparation.")
    
    elif choice == '4':
        file = find_latest_file('cv_results')
        if file:
            print(f"\nLoading: {file}")
            try:
                df = load_cv_results(file)
                print("\n" + "="*70)
                print("MODEL COMPARISON RESULTS")
                print("="*70)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
    
    elif choice == '5':
        file = find_latest_file('feature_importance')
        if file:
            print(f"\nLoading: {file}")
            try:
                df = load_feature_importance(file)
                print("\n" + "="*70)
                print("FEATURE IMPORTANCE (Top 20)")
                print("="*70)
                print(df.head(20).to_string(index=False))
                print("\n" + "="*70)
                print("LOW IMPORTANCE FEATURES (Bottom 10)")
                print("="*70)
                print(df.tail(10).to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
    
    elif choice == '6':
        print("Exiting...")
        return
    
    else:
        print("Invalid option")

def quick_plots():
    """Run all available plots from latest logs"""
    print("\n" + "="*70)
    print("GENERATING ALL PLOTS FROM LATEST RESULTS")
    print("="*70)
    
    # Plot 1: Learning curves
    lc_file = find_latest_file('learning_curve')
    if lc_file:
        print(f"\n[1/2] Plotting learning curves from: {os.path.basename(lc_file)}")
        try:
            plot_learning_curve_3d(log_file=lc_file)
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n[1/2] No learning curve data found")
    
    # Plot 2: Complexity analysis for top 3 models
    comp_file = find_latest_file('complexity_analysis_top3')
    if comp_file:
        print(f"\n[2/2] Plotting complexity analysis (top 3 models) from: {os.path.basename(comp_file)}")
        try:
            df = pd.read_csv(comp_file)
            models = df['model'].unique()
            for i, model_name in enumerate(models):
                model_data = df[df['model'] == model_name]
                complexities = model_data['complexity'].values
                train_errors = model_data['train_error'].values
                val_errors = model_data['val_error'].values
                print(f"  Plotting {model_name}...")
                plot_complexity_bias_variance(complexities, train_errors, val_errors)
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\n[2/2] No complexity analysis data found")
    
    print("\n" + "="*70)
    print("Plotting complete!")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick plot mode - plot all without prompting
        quick_plots()
    else:
        # Interactive menu mode
        while True:
            try:
                interactive_menu()
                print("\n" + "-"*70)
                cont = input("Continue? (y/n): ").strip().lower()
                if cont != 'y':
                    break
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
