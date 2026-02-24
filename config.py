"""
SQU: Sensitivity-Guided Quantum Machine Unlearning
===================================================

Configuration file for all experiments.

Author: Md Akmol Masud
Supervisor: Prof. Khalil
Institution: Jahangirnagar University
Target: QPAIN 2026 Conference
"""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Quantum Circuit Configuration
N_QUBITS = 4
N_LAYERS = 1  # Reduced from 2 for speed
LEARNING_RATE = 0.01

# Training Configuration
EPOCHS_QUANTUM = 20  # Increased for better convergence
EPOCHS_CLASSICAL = 50
BATCH_SIZE = 10

# Dataset Configuration - RIGOROUS EVALUATION
DATASETS = ['breast_cancer', 'mnist', 'fashion_mnist']  # All 3 datasets
TEST_SIZE = 0.3  # Standard 70/30 split
MAX_SAMPLES = 300  # Increased from 100 for statistical significance

# Unlearning Configuration
N_FORGET = 15  # Number of samples to forget (increased for better stats)
EPSILON_TOTAL = 1.0  # Total privacy budget

# SQU-specific: Sensitivity thresholds
SENSITIVITY_HIGH_THRESHOLD = 0.7
SENSITIVITY_MEDIUM_THRESHOLD = 0.3

# Noise levels for different sensitivity tiers (INCREASED for better privacy)
NOISE_HIGH_SENSITIVITY = 0.5  # Increased from 0.3
NOISE_MEDIUM_SENSITIVITY = 0.35  # Increased from 0.2
NOISE_LOW_SENSITIVITY = 0.2  # Increased from 0.1

# Output Configuration
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'

# Plotting Configuration
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
PLOT_STYLE = 'seaborn-v0_8-whitegrid'

# IEEE Paper Figure Sizes (inches)
FIGURE_SIZE_SINGLE_COLUMN = (3.5, 2.5)
FIGURE_SIZE_DOUBLE_COLUMN = (7.0, 3.0)

# Device Configuration
DEVICE = 'cpu'  # Force CPU for reproducibility

# Verbose Logging
VERBOSE = True
