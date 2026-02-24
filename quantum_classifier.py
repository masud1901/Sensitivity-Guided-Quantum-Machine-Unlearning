"""
Variational Quantum Circuit Classifier
======================================

PennyLane-based quantum classifier for binary classification.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.metrics import accuracy_score
import config

class QuantumClassifier:
    """Variational Quantum Circuit (VQC) for binary classification."""
    
    def __init__(self, n_qubits=None, n_layers=None, learning_rate=None):
        """
        Initialize quantum classifier.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimization
        """
        self.n_qubits = n_qubits or config.N_QUBITS
        self.n_layers = n_layers or config.N_LAYERS
        self.learning_rate = learning_rate or config.LEARNING_RATE
        
        # Quantum device (CPU-based)
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Initialize parameters
        self.params = pnp.random.random(
            (self.n_layers, self.n_qubits, 3), 
            requires_grad=True
        )
        
        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface='autograd')
        
        # Training history
        self.training_history = []
    
    def _circuit(self, params, x):
        """
        Quantum circuit: feature map + variational layers.
        
        Args:
            params: Circuit parameters
            x: Input features (angle-encoded)
        
        Returns:
            Expectation value of PauliZ on first qubit
        """
        # Angle encoding (feature map)
        for i in range(self.n_qubits):
            idx = i if i < len(x) else 0
            qml.RY(x[idx], wires=i)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates
            for qubit in range(self.n_qubits):
                qml.Rot(
                    params[layer, qubit, 0],
                    params[layer, qubit, 1],
                    params[layer, qubit, 2],
                    wires=qubit
                )
            
            # Entangling gates
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        
        return qml.expval(qml.PauliZ(0))
    
    def predict(self, X):
        """
        Predict raw outputs.
        
        Args:
            X: Input features
        
        Returns:
            Array of circuit outputs (expectation values)
        """
        return pnp.array([self.qnode(self.params, x) for x in X])
    
    def predict_classes(self, X, threshold=0.0):
        """
        Predict class labels.
        
        Args:
            X: Input features
            threshold: Classification threshold
        
        Returns:
            Binary predictions
        """
        return (self.predict(X) > threshold).astype(int)
    
    def _cost(self, params, X, y):
        """
        Cost function (MSE).
        
        Args:
            params: Circuit parameters
            X: Input features
            y: True labels
        
        Returns:
            Mean squared error
        """
        preds = pnp.array([self.qnode(params, x) for x in X])
        y_mapped = 2 * y - 1  # Map {0, 1} to {-1, 1}
        return pnp.mean((preds - y_mapped) ** 2)
    
    def train(self, X, y, epochs=None, batch_size=None, verbose=None):
        """
        Train the quantum classifier.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress
        
        Returns:
            Training history (list of losses)
        """
        epochs = epochs or config.EPOCHS_QUANTUM
        batch_size = batch_size or config.BATCH_SIZE
        verbose = verbose if verbose is not None else config.VERBOSE
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        n_samples = len(X)
        
        if verbose:
            print(f"Training VQC: {n_samples} samples, {epochs} epochs")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.params, loss = opt.step_and_cost(
                    lambda p: self._cost(p, X_batch, y_batch),
                    self.params
                )
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_history.append(float(avg_loss))
            
            if verbose and (epoch + 1) % 10 == 0:
                train_acc = accuracy_score(y, self.predict_classes(X))
                print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")
        
        if verbose:
            print(f"  Training complete! Final loss: {self.training_history[-1]:.4f}")
        
        return self.training_history
