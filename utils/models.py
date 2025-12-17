import numpy as np
import torch
import torch.nn as nn
from nystrom_attention import NystromAttention

def get_ML_clf(classifier, seed):
    if args.model == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1, max_iter=10000, random_state=seed, class_weight='balanced')

    elif classifier == "TabPFN":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)

    elif classifier == "RF":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=seed, class_weight='balanced')

    elif classifier == "SVM":
        from sklearn.svm import SVC
        clf = SVC(random_state=seed, class_weight='balanced', probability=True)

    elif classifier == "XGB":
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=seed, eval_metric="logloss", use_label_encoder=False)

    elif classifier == "MLP":
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="perceptron", max_iter=10000, random_state=seed, class_weight="balanced")
    return clf

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

class TransMIL(nn.Module):
    def __init__(self, n_classes, L, is_adv = False, n_classes_adv = None):
        super(TransMIL, self).__init__()
        self.n_classes = n_classes
        self.n_classes_adv = n_classes_adv
        self.pos_layer = PPEG(dim=L)
        self._fc1 = nn.Sequential(nn.Linear(int(L*2), L), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, L))
        self.layer1 = TransLayer(dim=L)
        self.layer2 = TransLayer(dim=L)
        self.norm = nn.LayerNorm(L)
        self.final_relu = torch.nn.ReLU()

        # Classifier initialization
        self.classifier = torch.nn.Sequential(torch.nn.Linear(L, self.n_classes))
        self.is_adv = is_adv
        if self.is_adv:
            self.classifier_adv = torch.nn.Sequential(torch.nn.Linear(L, self.n_classes_adv))

    def forward(self, h):
        # h = kwargs['dataframe'].float()  # [B, n, 1024]
        # h = self.backbone_extractor(x)
        # h = self._fc1(h)  # [B, n, 512]

        # ---->Pad
        h = h.reshape(1, h.shape[0], -1)
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
        # np.random.choice(np.arange(H), size=add_length, replace=False)

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]  # .squeeze(dim = 0)

        h = self.final_relu(h)

        # ---->Predict
        outputs = self.classifier(h).squeeze()  # [BS=1, n_classes]
        if not self.is_adv:
            return outputs, h
        else:
            outputs_adv = self.classifier_adv(h).squeeze()
            return outputs, outputs_adv, h

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim = 512, dropout = 0.1):  # voy a utilizar una VGG asi que la dimensión es 512
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = 8,
            num_landmarks = dim // 2,  # number of landmarks
            pinv_iterations = 6, # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True, # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = dropout
        ).cuda()

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)  # estudiar porque se utilizan estas convoluciones
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x