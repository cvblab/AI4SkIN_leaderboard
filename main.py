import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from utils.utils import set_random_seeds, plot_confmx, load_data, plot_figures, get_fmsi, get_hyperparams

parser = argparse.ArgumentParser(description="AI4SkIN leaderboard")
parser.add_argument('--folder', type=str)
parser.add_argument('--encoder', type=str)
parser.add_argument('--model', type=str, default=None, choices=["ABMIL", "TransMIL", "MISimpleShot", "LR", "TabPFN", "SVM", "RF", "XGB", "MLP"])
parser.add_argument('--get-fmsi', action='store_true')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

set_random_seeds(seed_value=args.seed)  # Reproducibility

run_name = f"{args.model}_{args.encoder}" # Run name definition
print(f"[INFO] Running experiment: {run_name}")
device = "cuda" if torch.cuda.is_available() else "cpu" # Device

X, Y, patients, centers, subtypes = load_data(encoder=args.encoder, folder=args.folder) # Data loading
n_classes = len(np.unique(Y)) # Number of classes
L = X[0].shape[-1] # Latent space length

get_fmsi(X, centers, subtypes, args.encoder) if args.get_fmsi else None # Calculate FM-SI and plot 2D TSNE

# Data partitioning (patient-level stratified k-fold cross validation)
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
list_cm_val = []
for fold, (train_index, val_index) in enumerate(kf.split(np.zeros(len(Y)), Y, patients)):
    print("################################# K = " + str(fold) + " #################################")
    run_name_k = f"{run_name}_k{fold}"

    X_train_k, X_val_k = [X[i] for i in train_index], [X[i] for i in val_index]
    Y_train_k, Y_val_k = Y[train_index], Y[val_index]

    if args.model in ["LR", "TabPFN", "MISimpleShot", "SVM", "RF", "XGB", "MLP"]:
        if len(X_train_k[0].shape) == 1: # WSI-level features
            X_train_k, X_val_k = np.stack(X_train_k), np.stack(X_val_k)
        else:
            X_train_k = np.stack([X_train_k_it.mean(0) for X_train_k_it in X_train_k])
            X_val_k = np.stack([X_train_k_it.mean(0) for X_train_k_it in X_val_k])
        Y_train_k, Y_val_k = np.stack(Y_train_k), np.stack(Y_val_k)

        # MISimpleShot: similarity-based classifier
        if args.model == "MISimpleShot":
            from utils.models import MISimpleShot
            prompt_classifier = MISimpleShot(X_train_k, Y_train_k)
            pred_val_k = np.argmax(X_val_k@prompt_classifier.T, axis=1)
        else:
            from utils.models import get_ML_clf
            clf = get_ML_clf(classifier=args.model, seed=args.seed)
            clf.fit(X=X_train_k, y=Y_train_k)
            pred_val_k = clf.predict(X=X_val_k)
        val_cm_k = confusion_matrix(Y_val_k, pred_val_k)

    else:
        from utils.trainer import train_model, validate_model # MIL trainers
        if len(X_train_k[0].shape) == 1:
            print("[ERROR]: MIL is not enabled for slide-FM", file=sys.stderr)
            sys.exit(1)

        # MIL model selection
        if args.model == "ABMIL":
            from utils.models import shABMIL
            model = shABMIL(n_classes=n_classes, L=L).to(device) # Attention-based MIL
        elif args.model == "TransMIL":
            from utils.models import TransMIL
            model = TransMIL(n_classes=n_classes, L=L).to(device) # Transformer-based MIL

        optimizer, criterion, scheduler = get_hyperparams(args.lr, Y_train_k, args.epochs, model)
        metrics = train_model(model, optimizer, criterion, scheduler, X_train_k, Y_train_k, X_val_k, Y_val_k, args.epochs, run_name_k)
        plot_figures(metrics, run_name_k, args.model, args.encoder)
        val_cm_k = validate_model(model, X_val_k, Y_val_k)
    plot_confmx(val_cm_k, run_name_k, model=args.model, encoder=args.encoder)
    list_cm_val.append(val_cm_k)
    torch.cuda.empty_cache()

cfmx_val = np.sum(np.stack(list_cm_val),axis=0)
plot_confmx(cfmx_val, run_name, model=args.model, encoder=args.encoder)