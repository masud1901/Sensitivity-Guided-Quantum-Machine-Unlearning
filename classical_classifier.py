"""
Classical Neural Network Baseline
=================================

PyTorch-based classical ML baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import config

class ClassicalNN(nn.Module):
    """PyTorch neural network for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=2):
        """
        Initialize classical neural network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (2 for binary classification)
        """
        super(ClassicalNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features (numpy array)
        
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features (numpy array)
        
        Returns:
            Predicted labels
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def train_classical_classifier(X_train, y_train, input_dim, epochs=None, verbose=None):
    """
    Train a classical neural network.
    
    Args:
        X_train: Training features
        y_train: Training labels
        input_dim: Input feature dimension
        epochs: Number of training epochs
        verbose: Print training progress
    
    Returns:
        Trained model
    """
    epochs = epochs or config.EPOCHS_CLASSICAL
    verbose = verbose if verbose is not None else config.VERBOSE
    
    # Create model
    model = ClassicalNN(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    if verbose:
        print(f"Training Classical NN: {len(X_train)} samples, {epochs} epochs")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(loader)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")
    
    if verbose:
        print("  Training complete!")
    
    return model
