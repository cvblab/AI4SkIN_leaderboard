import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


def plot_figures(train_acc_epoch,test_acc_epoch,train_loss_epoch, test_loss_epoch, run_name, model, encoder):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Plotting training and testing losses
    axs[0].plot(train_loss_epoch, label='Train Loss')
    axs[0].plot(test_loss_epoch, label='Test Loss')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plotting training and testing accuracies
    axs[1].plot(train_acc_epoch, label='Train ACC')
    axs[1].plot(test_acc_epoch, label='Test ACC')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()

    path_save = os.path.join("./local_data/results/", model, encoder)
    os.makedirs(path_save, exist_ok=True)
    plt.savefig(os.path.join(path_save, f"curves_{run_name}.png"))

def plot_confmx(conf_matrix, run_name, model, encoder):
    TP = np.diag(conf_matrix)
    FN = np.sum(conf_matrix, axis=1) - TP
    bal_acc = round(np.mean(TP / (TP + FN)),4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'BACC = {bal_acc}')
    plt.tight_layout()

    path_save = os.path.join("./local_data/results/", model, encoder)
    os.makedirs(path_save, exist_ok=True)
    plt.savefig(os.path.join(path_save, f"cfmx_{run_name}.png"))

def load_data(folder, feature_extractor):
    dataframe = pd.read_excel("AI4SkIN.xlsx")
    list_WSI = dataframe['WSI'].values
    labels = dataframe['GT'].values
    patients = dataframe['patient'].values
    center = dataframe['center'].values
    data = [np.load(os.path.join(folder, feature_extractor, file_name + ".npy")) for file_name in tqdm(list_WSI)]
    return data, labels, list_WSI, patients, center

def get_fmsi(X, centers, encoder):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # Mean pooling and dimensionality reduction
    X = np.stack([np.mean(X_it, axis=0) for X_it in X])
    tsne = TSNE(n_components=2, random_state=42)
    X_2D = tsne.fit_transform(X)

    # Plot 2D Visualization
    plt.figure(figsize=(8, 6))
    for it, center in enumerate(np.unique(centers)):
        indices = centers == center
        plt.scatter(X_2D[indices, 0], X_2D[indices, 1], label=center)
    plt.legend()
    plt.grid(True)

    # Calculate silhouette score
    fmsi = round(silhouette_score(X_2D, centers), 4)
    plt.title(f"FM-SI ({encoder}) = {fmsi:.4f}")
    plt.tight_layout()

    path_save = os.path.join("./local_data/tsne/")
    os.makedirs(path_save, exist_ok=True)
    plt.savefig(os.path.join(path_save, f"tsne_{encoder}.png"))
    exit()

def set_random_seeds(seed_value=42):
    # Set seed for NumPy
    np.random.seed(seed_value)

    # Set seed for Python's built-in random module
    random.seed(seed_value)

    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False