"""
SQU: Sensitivity-Guided Quantum Unlearning
===========================================

Core implementation of the SQU framework with sensitivity-based
privacy budget allocation.

This is the main contribution of the paper.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import config

class SQU:
    """
    Sensitivity-Guided Quantum Unlearning Framework.
    
    Key innovation: Dynamically allocates privacy budget based on
    per-sample gradient sensitivity, achieving better privacy-utility
    trade-offs than static approaches.
    """
    
    def __init__(self, quantum_classifier, epsilon_total=None):
        """
        Initialize SQU framework.
        
        Args:
            quantum_classifier: Trained quantum classifier
            epsilon_total: Total privacy budget
        """
        self.qc = quantum_classifier
        self.epsilon = epsilon_total or config.EPSILON_TOTAL
        self.sensitivity_scores = []
        self.budget_allocation = []
    
    def compute_sensitivity(self, x, y):
        """
        Compute gradient-based sensitivity for a single sample.
        
        Args:
            x: Input features
            y: True label
        
        Returns:
            Gradient magnitude (sensitivity score)
        """
        def loss_fn(params):
            """Loss for single sample."""
            pred = self.qc.qnode(params, x)
            y_mapped = 2 * y - 1
            return (pred - y_mapped) ** 2
        
        # Compute gradient
        grads = qml.grad(loss_fn)(self.qc.params)
        
        # Gradient magnitude = sensitivity
        sensitivity = float(pnp.linalg.norm(grads))
        
        return sensitivity
    
    def analyze_sensitivity(self, X_forget, y_forget, verbose=None):
        """
        Analyze sensitivity for all forget samples.
        
        Args:
            X_forget: Samples to forget
            y_forget: Labels for forget samples
            verbose: Print progress
        
        Returns:
            Array of normalized sensitivity scores
        """
        verbose = verbose if verbose is not None else config.VERBOSE
        
        if verbose:
            print(f"Analyzing sensitivity for {len(X_forget)} samples...")
        
        sensitivities = []
        for i, (x, y) in enumerate(zip(X_forget, y_forget)):
            sens = self.compute_sensitivity(x, y)
            sensitivities.append(sens)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  Sample {i+1}/{len(X_forget)}: sensitivity = {sens:.4f}")
        
        sensitivities = np.array(sensitivities)
        
        # Normalize to [0, 1]
        if sensitivities.max() > sensitivities.min():
            sens_norm = (sensitivities - sensitivities.min()) / \
                        (sensitivities.max() - sensitivities.min())
        else:
            sens_norm = np.ones_like(sensitivities)
        
        self.sensitivity_scores = sens_norm
        
        if verbose:
            print(f"  Sensitivity range: [{sensitivities.min():.4f}, {sensitivities.max():.4f}]")
            print(f"  Mean: {sensitivities.mean():.4f}, Std: {sensitivities.std():.4f}")
        
        return sens_norm
    
    def allocate_budget(self, verbose=None):
        """
        Allocate privacy budget based on sensitivity.
        
        Higher sensitivity → More privacy budget (stronger forgetting)
        
        Args:
            verbose: Print allocation details
        
        Returns:
            Array of per-sample budgets
        """
        verbose = verbose if verbose is not None else config.VERBOSE
        
        # Proportional allocation
        budgets = (self.sensitivity_scores / (self.sensitivity_scores.sum() + 1e-8)) * self.epsilon
        
        self.budget_allocation = budgets
        
        if verbose:
            print(f"Budget allocation:")
            print(f"  Min: {budgets.min():.4f}, Max: {budgets.max():.4f}")
            print(f"  Mean: {budgets.mean():.4f}, Total: {budgets.sum():.4f}")
        
        return budgets
    
    def unlearn(self, X_forget, y_forget, X_retain=None, y_retain=None, verbose=None):
        """
        Apply SQU unlearning with sensitivity-guided noise injection + fine-tuning.
        
        CRITICAL: Noise is injected BEFORE fine-tuning to follow DP principles.
        This masks the influence of forget samples, then fine-tuning recovers utility.
        
        Args:
            X_forget: Samples to forget
            y_forget: Labels for forget samples
            X_retain: Samples to retain (for fine-tuning)
            y_retain: Labels for retain samples
            verbose: Print progress
        
        Returns:
            Dictionary with unlearning statistics
        """
        verbose = verbose if verbose is not None else config.VERBOSE
        
        if verbose:
            print(f"\n{'='*60}")
            print("SQU: Sensitivity-Guided Quantum Unlearning")
            print(f"{'='*60}")
        
        # Step 1: Analyze sensitivity
        sens_norm = self.analyze_sensitivity(X_forget, y_forget, verbose)
        
        # Step 2: Allocate budget
        budgets = self.allocate_budget(verbose)
        
        # Step 3: Apply noise FIRST (based on sensitivity) - CRITICAL FIX
        if verbose:
            print(f"\nApplying sensitivity-guided noise (BEFORE fine-tuning)...")
        
        high_count = 0
        medium_count = 0
        low_count = 0
        
        for i, (sens, budget) in enumerate(zip(sens_norm, budgets)):
            # Determine noise level based on sensitivity
            if sens > config.SENSITIVITY_HIGH_THRESHOLD:
                noise_level = budget * config.NOISE_HIGH_SENSITIVITY
                tier = "HIGH"
                high_count += 1
            elif sens > config.SENSITIVITY_MEDIUM_THRESHOLD:
                noise_level = budget * config.NOISE_MEDIUM_SENSITIVITY
                tier = "MEDIUM"
                medium_count += 1
            else:
                noise_level = budget * config.NOISE_LOW_SENSITIVITY
                tier = "LOW"
                low_count += 1
            
            # Inject noise into parameters
            noise = pnp.random.normal(0, noise_level, self.qc.params.shape)
            self.qc.params = self.qc.params + noise
            
            if verbose and (i + 1) % 5 == 0:
                print(f"  Sample {i+1}/{len(X_forget)}: {tier} sensitivity, noise={noise_level:.4f}")
        
        # Step 4: Fine-tune on retain set AFTER noise (to recover utility)
        if X_retain is not None and y_retain is not None:
            if verbose:
                print(f"\nFine-tuning on retain set ({len(X_retain)} samples)...")
            # Fewer epochs to avoid overfitting after noise injection
            self.qc.train(X_retain, y_retain, epochs=5, verbose=False)
            if verbose:
                print(f"  Fine-tuning complete")
        
        # Summary statistics
        stats = {
            'high_sensitivity': high_count,
            'medium_sensitivity': medium_count,
            'low_sensitivity': low_count,
            'total_samples': len(X_forget),
            'total_budget': budgets.sum(),
            'sensitivity_scores': self.sensitivity_scores,
            'budget_allocation': self.budget_allocation
        }
        
        if verbose:
            print(f"\nUnlearning complete:")
            print(f"  High sensitivity samples: {high_count}")
            print(f"  Medium sensitivity samples: {medium_count}")
            print(f"  Low sensitivity samples: {low_count}")
            print(f"{'='*60}\n")
        
        return stats
    
    @staticmethod
    def unlearn_static(quantum_classifier, X_forget, y_forget, epsilon_total=None, verbose=None):
        """
        Baseline: Static unlearning (uniform budget allocation).
        
        Args:
            quantum_classifier: Trained quantum classifier
            X_forget: Samples to forget
            y_forget: Labels
            epsilon_total: Total privacy budget
            verbose: Print progress
        
        Returns:
            Dictionary with unlearning statistics
        """
        epsilon = epsilon_total or config.EPSILON_TOTAL
        verbose = verbose if verbose is not None else config.VERBOSE
        
        if verbose:
            print(f"\nBaseline: Static Unlearning (uniform budget)")
        
        # Uniform budget allocation
        n_samples = len(X_forget)
        budget_per_sample = epsilon / n_samples
        
        # Apply uniform noise
        total_noise_magnitude = 0
        for i in range(n_samples):
            noise_level = budget_per_sample * config.NOISE_MEDIUM_SENSITIVITY
            noise = pnp.random.normal(0, noise_level, quantum_classifier.params.shape)
            quantum_classifier.params = quantum_classifier.params + noise
            total_noise_magnitude += pnp.linalg.norm(noise)
        
        stats = {
            'method': 'static',
            'total_samples': n_samples,
            'budget_per_sample': budget_per_sample,
            'total_noise': float(total_noise_magnitude)
        }
        
        if verbose:
            print(f"  Applied uniform noise: {budget_per_sample:.4f} per sample")
            print(f"  Total noise magnitude: {total_noise_magnitude:.4f}\n")
        
        return stats
