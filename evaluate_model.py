import numpy as np
import matplotlib.pyplot as plt
from model import BPNN
from utils import create_folds

def evaluate_model(X, y, hidden_neurons, lr, k_folds=5, plot_loss=False):
    folds = create_folds(X, y, k_folds)
    fold_losses = []
    all_fold_loss_curves = []

    for i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = BPNN(input_size=X.shape[1], hidden_size=hidden_neurons,
                     output_size=1, learning_rate=lr)
        losses = model.train(X_train, y_train, epochs=1000)
        all_fold_loss_curves.append(losses)

        preds = model.predict(X_test)
        loss = np.mean((y_test - preds) ** 2)
        fold_losses.append(loss)

        print(f"Fold {i+1}/{k_folds} | MSE: {loss:.6f}")

    avg_loss = np.mean(fold_losses)
    print(f"\nAverage {k_folds}-Fold Loss = {avg_loss:.6f}")

    if plot_loss:
        plt.figure(figsize=(8, 5))
        for i, losses in enumerate(all_fold_loss_curves):
            plt.plot(losses, alpha=0.4, label=f"Fold {i+1}")
        max_len = min(len(l) for l in all_fold_loss_curves)
        mean_loss = np.mean([l[:max_len] for l in all_fold_loss_curves], axis=0)
        plt.plot(mean_loss, color='black', linewidth=2, label="Mean Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss (MSE)")
        plt.title(f"Loss Curves Across {k_folds} Folds (Hidden={hidden_neurons}, LR={lr})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return avg_loss
