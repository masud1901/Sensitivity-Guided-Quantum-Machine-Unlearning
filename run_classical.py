import numpy as np
from datasets import load_dataset, prepare_classical_data
from classical_classifier import train_classical_classifier
from attacks import MembershipInferenceAttack
from sklearn.metrics import accuracy_score
import config

for dataset_name in config.DATASETS:
    print(f"\n--- {dataset_name} ---")
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = prepare_classical_data(X, y)
    
    n_forget = min(config.N_FORGET, len(X_train) // 4)
    forget_idx = np.random.choice(len(X_train), size=n_forget, replace=False)
    X_forget = X_train[forget_idx]
    
    # Train Classical NN
    model = train_classical_classifier(X_train, y_train, input_dim=X_train.shape[1], epochs=50, verbose=False)
    
    # Test accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Classical Test Accuracy: {acc:.4f}")
    
    # MIA
    attacker = MembershipInferenceAttack()
    # Modify attack to accept classical model
    try:
        attack_auc = attacker.attack_classical(model, X_forget, X_test)
        print(f"Classical MIA AUC: {attack_auc:.4f}")
    except AttributeError:
        print("MIA Attack for classical not implemented yet.")
