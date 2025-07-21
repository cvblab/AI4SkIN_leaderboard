import torch
import numpy as np

def MISimpleShot(X_train_k, Y_train_k):
    prompt_classifier = []
    for class_idx in np.unique(Y_train_k):
        class_wsi = X_train_k[Y_train_k == class_idx]
        prompt_class = np.mean(class_wsi, axis=0)  # Prototype
        prompt_class = prompt_class / np.linalg.norm(prompt_class, ord=2, axis=-1, keepdims=True)
        prompt_classifier.append(prompt_class)
    prompt_classifier = np.stack(prompt_classifier)
    return prompt_classifier

class ABMIL(torch.nn.Module):
    def __init__(self, L, p=0.25):
        super(ABMIL, self).__init__()

        # Attention MIL embedding from Ilse et al. (2018) for MIL.
        # Class based on Julio Silva's MILAggregation class in PyTorch
        self.L = L
        self.D = int(self.L/4)
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh(),
            torch.nn.Dropout(p)
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p)
        )

        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        A_V = self.attention_V(features)  # Attention
        A_U = self.attention_U(features)  # Gate
        w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)
        features = torch.transpose(features, 1, 0)
        embedding = torch.squeeze(torch.mm(features, w))  # MIL Attention
        return embedding, w.squeeze(dim=-1)

class shABMIL(torch.nn.Module):
    def __init__(self, n_classes, L, p=0.25):
        super(shABMIL, self).__init__()
        self.ABMIL = ABMIL(L, p)
        self.classifier = torch.nn.Linear(L,n_classes)

    def forward(self, features):
        embedding, w = self.ABMIL(features)
        output = self.classifier(embedding)
        return output, embedding