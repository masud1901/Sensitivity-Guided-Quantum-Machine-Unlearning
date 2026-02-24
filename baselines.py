"""
Additional Baseline Unlearning Methods
=======================================

Implements additional baseline methods for comprehensive comparison:
1. Simple Noise Injection (no sensitivity)
2. Gradient Ascent Unlearning
3. Retrain from Scratch (gold standard)
"""

import numpy as np
from pennylane import numpy as pnp
import config

def simple_noise_baseline(quantum_classifier, X_forget, epsilon_total=None, verbose=None):
    """
    Baseline 1: Simple random noise injection (no sensitivity analysis).
    
    Adds Gaussian noise to all parameters uniformly.
    
    Args:
        quantum_classifier: Trained quantum classifier
        X_forget: Forget samples (not used, kept for API consistency)
        epsilon_total: Total privacy budget
        verbose: Print progress
    
    Returns:
        Dictionary with statistics
    """
    epsilon = epsilon_total or config.EPSILON_TOTAL
    verbose = verbose if verbose is not None else config.VERBOSE
    
    if verbose:
        print(f"\nBaseline: Simple Noise Injection")
    
    # Single noise injection to all parameters
    noise_std = epsilon * 0.3  # Similar scale to SQU
    noise = pnp.random.normal(0, noise_std, quantum_classifier.params.shape)
    quantum_classifier.params = quantum_classifier.params + noise
    
    stats = {
        'method': 'simple_noise',
        'noise_std': noise_std,
        'total_noise': float(pnp.linalg.norm(noise))
    }
    
    if verbose:
        print(f"  Applied noise with std={noise_std:.4f}")
        print(f"  Total noise magnitude: {stats['total_noise']:.4f}\n")
    
    return stats

def gradient_ascent_unlearning(quantum_classifier, X_forget, y_forget, 
                                steps=5, learning_rate=0.01, verbose=None):
    """
    Baseline 2: Gradient Ascent Unlearning.
    
    Performs gradient ascent on forget samples to maximize loss,
    effectively "pushing away" from memorized patterns.
    
    Args:
        quantum_classifier: Trained quantum classifier
        X_forget: Samples to forget
        y_forget: Labels for forget samples
        steps: Number of gradient ascent steps
        learning_rate: Learning rate for ascent
        verbose: Print progress
    
    Returns:
        Dictionary with statistics
    """
    verbose = verbose if verbose is not None else config.VERBOSE
    
    if verbose:
        print(f"\nBaseline: Gradient Ascent Unlearning ({steps} steps)")
    
    import pennylane as qml
    
    initial_loss = 0.0
    for x, y in zip(X_forget, y_forget):
        pred = quantum_classifier.qnode(quantum_classifier.params, x)
        y_mapped = 2 * y - 1
        initial_loss += (pred - y_mapped) ** 2
    initial_loss /= len(X_forget)
    
    for step in range(steps):
        # Compute gradient for forget set
        def forget_loss(params):
            loss = 0.0
            for x, y in zip(X_forget, y_forget):
                pred = quantum_classifier.qnode(params, x)
                y_mapped = 2 * y - 1
                loss += (pred - y_mapped) ** 2
            return loss / len(X_forget)
        
        grad = qml.grad(forget_loss)(quantum_classifier.params)
        
        # Gradient ASCENT (increase loss on forget samples)
        quantum_classifier.params = quantum_classifier.params + learning_rate * grad
        
        if verbose and (step + 1) % 2 == 0:
            current_loss = forget_loss(quantum_classifier.params)
            print(f"  Step {step+1}/{steps}: Forget loss = {current_loss:.4f}")
    
    final_loss = forget_loss(quantum_classifier.params)
    
    stats = {
        'method': 'gradient_ascent',
        'steps': steps,
        'learning_rate': learning_rate,
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'loss_increase': float(final_loss - initial_loss)
    }
    
    if verbose:
        print(f"  Loss increased: {initial_loss:.4f} → {final_loss:.4f}\n")
    
    return stats
