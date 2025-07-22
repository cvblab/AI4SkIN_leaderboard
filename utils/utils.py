import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def plot_figures(metrics, run_name, model, encoder):
    train_acc_epoch, test_acc_epoch, train_loss_epoch, test_loss_epoch = metrics
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

def load_data(folder, encoder):
    dataframe = pd.read_excel("AI4SkIN_df.xlsx")
    list_WSI = dataframe['WSI'].values
    labels = dataframe['GT'].values
    subtypes = dataframe['subtype'].values
    patients = dataframe['patient'].values
    centers = dataframe['center'].values

    if not os.path.exists(os.path.join(folder, encoder)):
        import sys
        print(f"[ERROR] Encoder {encoder} is not available", file=sys.stderr)
        sys.exit(1)

    data = [np.load(os.path.join(folder, encoder, file_name + ".npy")) for file_name in tqdm(list_WSI)]
    return data, labels, patients, centers, subtypes

def get_hyperparams(lr, Y_train_k, epochs, model):
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train_k), y=Y_train_k)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    return optimizer, criterion, scheduler

def get_ri(X,centers,labels,k=20):
    sim_mx = cosine_similarity(X)
    num, den = 0.0, 0.0
    for sim_mx_it, labels_it, centers_it in zip(sim_mx, labels, centers):
        idx = np.argsort(sim_mx_it)[::-1][1:k + 1]
        label, center = labels[idx], centers[idx]
        num += np.sum(label == labels_it)
        den += np.sum(center == centers_it)
    ri = num / den
    return ri

def get_fmsi(X, centers, labels, encoder):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # Mean pooling and dimensionality reduction
    X = np.stack(X) if len(X[0].shape) == 1 else np.stack([np.mean(x, axis=0) for x in X])
    tsne = TSNE(n_components=2, random_state=42)
    X_2D = tsne.fit_transform(X)

    # Plot 2D Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    identifiers = [labels, centers]
    for ax, id_it in zip(axs, identifiers):
        for label in np.unique(id_it):
            indices = id_it == label
            ax.scatter(X_2D[indices, 0], X_2D[indices, 1], label=str(label))
        ax.legend()
        ax.grid(True, zorder=0)

    fmsi = silhouette_score(X_2D, centers) # Calculate silhouette score
    ri = get_ri(X, centers, labels) # Calculate robusness index (RI)
    axs[1].set_title(f"{encoder}: FM-SI = {fmsi:.4f} | RI = {ri:.4f}")
    plt.tight_layout()

    path_save = os.path.join("./local_data/tsne/")
    os.makedirs(path_save, exist_ok=True)
    plt.savefig(os.path.join(path_save, f"tsne_{encoder}.png"))
    print("[INFO] 2D t-SNE plot saved")
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