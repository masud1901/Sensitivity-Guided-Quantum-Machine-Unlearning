<p align="center">
  <img src="../figs/vqc_circuit.pdf" alt="VQC Circuit" width="600"/>
</p>

<h1 align="center">SQU: Sensitivity-Guided Quantum Unlearning</h1>

<p align="center">
  <b>Privacy-Preserving Machine Unlearning for Variational Quantum Circuits</b>
</p>

<p align="center">
  <a href="https://qpain.org/"><img src="https://img.shields.io/badge/QPAIN_2026-Accepted_(Poster)-brightgreen" alt="QPAIN 2026"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://pennylane.ai/"><img src="https://img.shields.io/badge/PennyLane-0.35+-purple.svg" alt="PennyLane"></a>
</p>

---

Official implementation of the paper:

> **Sensitivity-Guided Quantum Machine Unlearning for Privacy-Preserving ML**
>
> Md. Akmol Masud, Zannat Hossain Tamim, Md. Musa Kalimullah Ratul, and Md. Biplob Hosen
>
> *IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN), 2026*

---

## Overview

SQU is a novel quantum machine unlearning framework that enables **post-training data removal** from Variational Quantum Circuits (VQCs), addressing the Right to Be Forgotten (RTBF) requirements under GDPR.

**Key Innovation**: Unlike classical unlearning methods that apply uniform noise, SQU computes **per-sample gradient sensitivity** and dynamically allocates privacy budgets — applying stronger forgetting to high-sensitivity samples while preserving model utility.

### How SQU Works

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│    Data      │    │  VQC Training &  │    │   Sensitivity-  │    │   Fine-tune  │
│ Preprocessing│───▶│  Forget Request  │───▶│  Guided Noise   │───▶│  on Retain   │
│ (PCA, Encode)│    │  (Split D_f/D_r) │    │  Injection      │    │  Set (D_r)   │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
                                                   │
                                           ┌───────┴───────┐
                                           │ Per-sample     │
                                           │ sensitivity    │
                                           │ → adaptive σ_k │
                                           └───────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/masud1901/Sensitivity-Guided-Quantum-Machine-Unlearning.git
cd Sensitivity-Guided-Quantum-Machine-Unlearning

# Option 1: Using setup script
chmod +x setup.sh
./setup.sh

# Option 2: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Experiments

```bash
# Activate environment
source venv/bin/activate

# Run full experimental suite (all 3 datasets)
python run_experiments.py

# Generate publication-ready figures
python generate_figures.py
```

**Expected runtime**: ~20-30 minutes on CPU

---

## Project Structure

```
SQU/
├── squ.py                    # 🔑 Core SQU framework (main contribution)
│                             #    - Gradient-based sensitivity analysis
│                             #    - Adaptive budget allocation
│                             #    - Noise injection + fine-tuning
├── quantum_classifier.py     # VQC implementation (PennyLane)
├── classical_classifier.py   # PyTorch classical baseline
├── attacks.py                # Membership Inference Attack (MIA)
├── datasets.py               # Data loading (Breast Cancer, MNIST, Fashion-MNIST)
├── config.py                 # Hyperparameters and configuration
├── run_experiments.py        # Main experiment runner
├── run_classical.py          # Classical NN baseline comparison
├── generate_figures.py       # Publication figure generation
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-click setup script
└── results/                  # Experimental outputs
    ├── squ_results_*.csv     # Raw results data
    └── figures/              # Generated figures (PNG, 300 DPI)
```

---

## Key Results

SQU is evaluated on three benchmark datasets against static unlearning and retrain-from-scratch baselines:

| Dataset | SQU Accuracy | Retrain Accuracy | SQU MIA AUC | Privacy Gain |
|:---|:---:|:---:|:---:|:---:|
| **Breast Cancer** | 83.3% | 84.4% | **0.648** | **+0.67%** |
| **MNIST** | 97.9% | 97.9% | 0.530 | 0.00% |
| **Fashion-MNIST** | 79.2% | 79.2% | 0.609 | −0.08% |

- ✅ SQU **matches retrain-from-scratch accuracy** across all datasets
- ✅ **Privacy improvement** on Breast Cancer (+0.67% MIA AUC reduction)
- ✅ Avoids the **privacy degradation** seen with static unlearning on MNIST (−7.19%)

### Classical vs. Quantum Comparison

To contextualize VQC performance, we compare against a classical Neural Network trained on the same data for the same number of epochs (20):

| Property | Classical NN | Quantum VQC |
|:---|:---:|:---:|
| **Trainable parameters** | ~2,498 | **12** |
| **Accuracy (Breast Cancer)** | 94.4% | 84.4% |
| **Exact per-sample sensitivity** | Intractable | ✅ **Feasible** |
| **Unlearning parameter coverage** | Partial | ✅ **Complete** |

> **Key insight**: The VQC's 200× fewer parameters enable *exact, complete* sensitivity analysis for every forget sample — an advantage that is computationally infeasible at classical scale. This compactness is the structural advantage that makes SQU possible.

```bash
# Run classical baseline comparison
python run_classical.py
```

---

## Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|:---|:---:|:---|
| `N_QUBITS` | 4 | Number of qubits in VQC |
| `N_LAYERS` | 1 | Variational layers |
| `EPOCHS_QUANTUM` | 20 | Training epochs |
| `LEARNING_RATE` | 0.01 | Optimizer learning rate |
| `N_FORGET` | 15 | Samples to forget |
| `EPSILON_TOTAL` | 1.0 | Total privacy budget (ε) |
| `MAX_SAMPLES` | 300 | Dataset size |

---

## Dependencies

- Python ≥ 3.9
- [PennyLane](https://pennylane.ai/) ≥ 0.35 — Quantum ML framework
- [PyTorch](https://pytorch.org/) ≥ 2.0 — Classical baselines (CPU)
- [scikit-learn](https://scikit-learn.org/) ≥ 1.3 — Datasets & metrics
- NumPy, Pandas, Matplotlib, Seaborn

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{masud2026squ,
  title     = {Sensitivity-Guided Quantum Machine Unlearning for Privacy-Preserving {ML}},
  author    = {Masud, Md. Akmol and Tamim, Zannat Hossain and Ratul, Md. Musa Kalimullah and Hosen, Md. Biplob},
  booktitle = {2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
