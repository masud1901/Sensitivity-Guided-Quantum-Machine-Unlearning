"""
Membership Inference Attack
============================

Implementation of membership inference attacks for privacy evaluation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
import config

class MembershipInferenceAttack:
    """Membership Inference Attack using prediction confidence."""
    
    def attack_quantum(self, qc, X_member, X_nonmember, verbose=None):
        """
        Attack quantum model.
        
        Args:
            qc: Quantum classifier
            X_member: Member samples (training data)
            X_nonmember: Non-member samples (test data)
            verbose: Print attack details
        
        Returns:
            Attack AUC score (higher = more privacy leakage)
        """
        verbose = verbose if verbose is not None else config.VERBOSE
        
        # Get confidence scores (magnitude of prediction)
        conf_member = np.abs(qc.predict(X_member))
        conf_nonmember = np.abs(qc.predict(X_nonmember))
        
        # Prepare for AUC calculation
        y_true = np.concatenate([
            np.ones(len(conf_member)),   # Members = 1
            np.zeros(len(conf_nonmember))  # Non-members = 0
        ])
        y_scores = np.concatenate([conf_member, conf_nonmember])
        
        # Calculate AUC (0.5 = random guessing, 1.0 = perfect attack)
        raw_auc = roc_auc_score(y_true, y_scores)
        
        # IMPORTANT: In security literature, AUC < 0.5 means anti-correlation
        # A smart attacker flips predictions, so effective AUC = max(AUC, 1-AUC)
        effective_auc = max(raw_auc, 1 - raw_auc)
        
        if verbose:
            print(f"  Member confidence (avg): {np.mean(conf_member):.4f}")
            print(f"  Non-member confidence (avg): {np.mean(conf_nonmember):.4f}")
            print(f"  Attack AUC (raw): {raw_auc:.4f}")
            print(f"  Attack AUC (effective): {effective_auc:.4f} (0.5=random, 1.0=perfect)")
        
        return float(effective_auc)
    
    def attack_classical(self, model, X_member, X_nonmember, verbose=None):
        """
        Attack classical model.
        
        Args:
            model: Classical classifier
            X_member: Member samples (training data)
            X_nonmember: Non-member samples (test data)
            verbose: Print attack details
        
        Returns:
            Attack AUC score
        """
        verbose = verbose if verbose is not None else config.VERBOSE
        
        # Get max probability as confidence
        conf_member = np.max(model.predict_proba(X_member), axis=1)
        conf_nonmember = np.max(model.predict_proba(X_nonmember), axis=1)
        
        # Prepare for AUC calculation
        y_true = np.concatenate([
            np.ones(len(conf_member)),
            np.zeros(len(conf_nonmember))
        ])
        y_scores = np.concatenate([conf_member, conf_nonmember])
        
        # Calculate AUC
        raw_auc = roc_auc_score(y_true, y_scores)
        effective_auc = max(raw_auc, 1 - raw_auc)
        
        if verbose:
            print(f"  Member confidence (avg): {np.mean(conf_member):.4f}")
            print(f"  Non-member confidence (avg): {np.mean(conf_nonmember):.4f}")
            print(f"  Attack AUC (effective): {effective_auc:.4f}")
        
        return float(effective_auc)
