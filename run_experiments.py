"""
Main Experiment Runner
=======================

Runs complete experimental suite for QPAIN 2026 paper.

Experiments:
1. Train quantum + classical baselines
2. Run membership inference attacks (BEFORE unlearning)
3. Apply SQU unlearning
4. Apply static unlearning (baseline)
5. Run attacks again (AFTER unlearning)
6. Compare and export results
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

import config
from datasets import load_dataset, prepare_quantum_data, prepare_classical_data
from quantum_classifier import QuantumClassifier
from classical_classifier import train_classical_classifier
from attacks import MembershipInferenceAttack
from squ import SQU
from baselines import simple_noise_baseline, gradient_ascent_unlearning
from sklearn.metrics import accuracy_score

def create_output_dirs():
    """Create output directories."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

def run_single_experiment(dataset_name):
    """
    Run complete experiment on a single dataset.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*70}\n")
    
    # Load and prepare data
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = prepare_quantum_data(X, y)
    
    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    
    # Select forget samples
    n_forget = min(config.N_FORGET, len(X_train) // 4)
    forget_idx = np.random.choice(len(X_train), size=n_forget, replace=False)
    keep_mask = np.ones(len(X_train), dtype=bool)
    keep_mask[forget_idx] = False
    
    X_retain = X_train[keep_mask]
    y_retain = y_train[keep_mask]
    X_forget = X_train[forget_idx]
    y_forget = y_train[forget_idx]
    
    print(f"Forget: {n_forget} samples, Retain: {len(X_retain)} samples\n")
    
    # ========================================
    # BASELINE: Train quantum classifier
    # ========================================
    print("Training Quantum Classifier...")
    qc = QuantumClassifier()
    training_history = qc.train(X_train, y_train)  # Capture training history
    
    acc_before = accuracy_score(y_test, qc.predict_classes(X_test))
    print(f"  Test Accuracy: {acc_before:.4f}\n")
    
    # ========================================
    # ATTACK BEFORE UNLEARNING
    # ========================================
    print("Membership Inference Attack (BEFORE unlearning)...")
    attacker = MembershipInferenceAttack()
    attack_before = attacker.attack_quantum(qc, X_forget, X_test)
    print()
    
    # ========================================
    # SQU UNLEARNING (Our method)
    # ========================================
    print("Applying SQU (Sensitivity-Guided Unlearning)...")
    qc_squ = QuantumClassifier()
    qc_squ.params = qc.params.copy()  # Start from trained model
    
    squ = SQU(qc_squ)
    squ_stats = squ.unlearn(X_forget, y_forget, X_retain, y_retain)  # Pass retain set!
    
    acc_after_squ = accuracy_score(y_test, qc_squ.predict_classes(X_test))
    print(f"  Test Accuracy after SQU: {acc_after_squ:.4f}")
    
    print("\nMembership Inference Attack (AFTER SQU)...")
    attack_after_squ = attacker.attack_quantum(qc_squ, X_forget, X_test)
    print()
    
    # ========================================
    # STATIC UNLEARNING (Baseline)
    # ========================================
    print("Applying Static Unlearning (uniform budget)...")
    qc_static = QuantumClassifier()
    qc_static.params = qc.params.copy()  # Start from trained model
    
    static_stats = SQU.unlearn_static(qc_static, X_forget, y_forget)
    
    acc_after_static = accuracy_score(y_test, qc_static.predict_classes(X_test))
    print(f"  Test Accuracy after Static: {acc_after_static:.4f}")
    
    print("\nMembership Inference Attack (AFTER Static)...")
    attack_after_static = attacker.attack_quantum(qc_static, X_forget, X_test)
    
    # ========================================
    # SIMPLE NOISE BASELINE
    # ========================================
    print("Applying Simple Noise Baseline...")
    qc_simple = QuantumClassifier()
    qc_simple.params = qc.params.copy()
    
    simple_stats = simple_noise_baseline(qc_simple, X_forget)
    
    acc_after_simple = accuracy_score(y_test, qc_simple.predict_classes(X_test))
    print(f"  Test Accuracy after Simple Noise: {acc_after_simple:.4f}")
    
    print("\nMembership Inference Attack (AFTER Simple Noise)...")
    attack_after_simple = attacker.attack_quantum(qc_simple, X_forget, X_test)
    print()
    
    # ========================================
    # GRADIENT ASCENT BASELINE
    # ========================================
    print("Applying Gradient Ascent Unlearning...")
    qc_gradascent = QuantumClassifier()
    qc_gradascent.params = qc.params.copy()
    
    gradascent_stats = gradient_ascent_unlearning(qc_gradascent, X_forget, y_forget, steps=5)
    
    acc_after_gradascent = accuracy_score(y_test, qc_gradascent.predict_classes(X_test))
    print(f"  Test Accuracy after Gradient Ascent: {acc_after_gradascent:.4f}")
    
    print("\nMembership Inference Attack (AFTER Gradient Ascent)...")
    attack_after_gradascent = attacker.attack_quantum(qc_gradascent, X_forget, X_test)
    print()
    
    # ========================================
    # RETRAIN FROM SCRATCH (Gold Standard)
    # ========================================
    print("Retraining from Scratch on Retain Set (Gold Standard)...")
    qc_retrain = QuantumClassifier()
    retrain_history = qc_retrain.train(X_retain, y_retain)  # Train ONLY on retain set
    
    acc_after_retrain = accuracy_score(y_test, qc_retrain.predict_classes(X_test))
    print(f"  Test Accuracy after Retrain: {acc_after_retrain:.4f}")
    
    print("\nMembership Inference Attack (AFTER Retrain)...")
    attack_after_retrain = attacker.attack_quantum(qc_retrain, X_forget, X_test)
    print()
    
    # ========================================
    # COMPILE RESULTS
    # ========================================
    results = {
        'dataset': dataset_name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_forget': n_forget,
        
        # Accuracy - all methods
        'acc_before': acc_before,
        'acc_after_squ': acc_after_squ,
        'acc_after_static': acc_after_static,
        'acc_after_simple': acc_after_simple,
        'acc_after_gradascent': acc_after_gradascent,
        'acc_after_retrain': acc_after_retrain,  # Gold standard
        
        # Accuracy drops
        'acc_drop_squ': acc_before - acc_after_squ,
        'acc_drop_static': acc_before - acc_after_static,
        'acc_drop_simple': acc_before - acc_after_simple,
        'acc_drop_gradascent': acc_before - acc_after_gradascent,
        'acc_drop_retrain': acc_before - acc_after_retrain,
        
        # Privacy (Attack AUC - effective)
        'attack_before': attack_before,
        'attack_after_squ': attack_after_squ,
        'attack_after_static': attack_after_static,
        'attack_after_simple': attack_after_simple,
        'attack_after_gradascent': attack_after_gradascent,
        'attack_after_retrain': attack_after_retrain,  # Gold standard
        
        # Privacy gains (higher = better)
        'privacy_gain_squ': attack_before - attack_after_squ,
        'privacy_gain_static': attack_before - attack_after_static,
        'privacy_gain_simple': attack_before - attack_after_simple,
        'privacy_gain_gradascent': attack_before - attack_after_gradascent,
        'privacy_gain_retrain': attack_before - attack_after_retrain,
        
        # SQU-specific
        'squ_high_sens': squ_stats['high_sensitivity'],
        'squ_medium_sens': squ_stats['medium_sensitivity'],
        'squ_low_sens': squ_stats['low_sensitivity'],
    }
    
    return results, training_history

def run_all_experiments():
    """Run experiments on all datasets."""
    print("\n" + "="*70)
    print("SQU: Sensitivity-Guided Quantum Machine Unlearning")
    print("QPAIN 2026 Conference Paper - Experimental Suite")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {', '.join(config.DATASETS)}")
    print(f"Qubits: {config.N_QUBITS}, Layers: {config.N_LAYERS}")
    print("="*70)
    
    create_output_dirs()
    
    all_results = []
    all_training_histories = []
    
    for dataset_name in config.DATASETS:
        try:
            results, training_history = run_single_experiment(dataset_name)
            all_results.append(results)
            all_training_histories.append(training_history)
        except Exception as e:
            print(f"\nERROR on {dataset_name}: {e}")
            continue
    
    # ========================================
    # EXPORT RESULTS
    # ========================================
    df = pd.DataFrame(all_results)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(config.RESULTS_DIR, f'squ_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # Save training history for visualization
    import json
    if all_training_histories:
        training_path = csv_path.replace('.csv', '_training.json')
        with open(training_path, 'w') as f:
            json.dump({'training_history': all_training_histories[0]}, f)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary
    print("\nSUMMARY TABLE:")
    print(df[['dataset', 'acc_before', 'acc_after_squ', 'attack_before', 'attack_after_squ']].to_string(index=False))
    
    print("\n" + "="*70)
    print("Next step: Run generate_figures.py to create publication plots")
    print("="*70 + "\n")
    
    return df

if __name__ == "__main__":
    results_df = run_all_experiments()
