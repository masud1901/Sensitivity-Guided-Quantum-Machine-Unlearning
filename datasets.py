"""
Dataset Loading and Preprocessing
==================================

Handles loading MNIST, Fashion-MNIST, and Breast Cancer datasets
with preprocessing for both classical and quantum models.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

def load_mnist(n_samples=800):
    """Load MNIST dataset (binary: 0 vs 1)."""
    try:
        from tensorflow import keras
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Combine and flatten
        X = np.concatenate([X_train, X_test]).reshape(-1, 784)
        y = np.concatenate([y_train, y_test])
        
        # Binary classification: 0 vs 1
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        
        # Subsample
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[indices].astype(float), y[indices]
        y = (y == 1).astype(int)
        
        return X, y
    except ImportError:
        print("Warning: TensorFlow not available, using sklearn fetch_openml")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[indices], y[indices]
        y = (y == 1).astype(int)
        return X, y

def load_fashion_mnist(n_samples=800):
    """Load Fashion-MNIST dataset (binary: T-shirt vs Trouser)."""
    try:
        from tensorflow import keras
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        
        # Combine and flatten
        X = np.concatenate([X_train, X_test]).reshape(-1, 784)
        y = np.concatenate([y_train, y_test])
        
        # Binary: T-shirt (0) vs Trouser (1)
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        
        # Subsample
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[indices].astype(float), y[indices]
        y = (y == 1).astype(int)
        
        return X, y
    except ImportError:
        print("Warning: TensorFlow not available, using sklearn fetch_openml")
        from sklearn.datasets import fetch_openml
        fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
        X, y = fmnist.data, fmnist.target.astype(int)
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[indices], y[indices]
        y = (y == 1).astype(int)
        return X, y

def load_breast_cancer_data(n_samples=None):
    """Load Breast Cancer Wisconsin dataset."""
    if n_samples is None:
        n_samples = getattr(__import__('config'), 'MAX_SAMPLES', 400)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Subsample
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X, y = X[indices], y[indices]
    
    return X, y

def load_dataset(name):
    """
    Load a dataset by name.
    
    Args:
        name: One of ['mnist', 'fashion_mnist', 'breast_cancer']
    
    Returns:
        X, y: Features and labels
    """
    if name == 'mnist':
        return load_mnist()
    elif name == 'fashion_mnist':
        return load_fashion_mnist()
    elif name == 'breast_cancer':
        return load_breast_cancer_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def prepare_classical_data(X, y):
    """
    Prepare data for classical ML models.
    
    Args:
        X: Raw features
        y: Labels
    
    Returns:
        X_train, X_test, y_train, y_test: Scaled and split data
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def prepare_quantum_data(X, y, n_qubits=None):
    """
    Prepare data for quantum ML models.
    
    Args:
        X: Raw features
        y: Labels
        n_qubits: Number of qubits (defaults to config.N_QUBITS)
    
    Returns:
        X_train, X_test, y_train, y_test: PCA-reduced, angle-encoded, and split data
    """
    if n_qubits is None:
        n_qubits = config.N_QUBITS
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA dimensionality reduction
    n_components = min(n_qubits, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Normalize to [-π/2, π/2] for angle encoding
    X_quantum = np.pi * (X_pca - X_pca.min()) / (X_pca.max() - X_pca.min() + 1e-8) - np.pi/2
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test
