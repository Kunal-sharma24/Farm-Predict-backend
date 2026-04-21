import torch
import torch.nn as nn
import torch.nn.functional as F

class DNFNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_trees, n_classes):
        super().__init__()
        self.trees = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_trees)])
        self.output = nn.Linear(hidden_dim * n_trees, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        tree_outputs = [F.relu(tree(x)) for tree in self.trees]
        h = torch.cat(tree_outputs, dim=-1)
        return self.output(self.dropout(h))


class AutoInt(nn.Module):
    def __init__(self, in_dim, n_heads, n_layers, embed_dim, n_classes):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        for attn in self.attn_layers:
            x, _ = attn(x, x, x)
        return self.fc(x.squeeze(0))


class GrowNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_stages, n_classes):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + (n_classes if i > 0 else 0), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes)
            ) for i in range(n_stages)
        ])

    def forward(self, x):
        prev = None
        outputs = []
        for i, stage in enumerate(self.stages):
            out = stage(x if i == 0 else torch.cat([x, prev], dim=-1))
            outputs.append(out)
            prev = out
        return sum(outputs) / len(outputs)


class SAINT(nn.Module):
    def __init__(self, in_dim, embed_dim, n_heads, n_layers, n_classes):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        for attn in self.attn:
            x, _ = attn(x, x, x)
        return self.fc(x.squeeze(0))


class NAM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(in_dim)
        ])
        self.output = nn.Linear(hidden_dim * in_dim, n_classes)

    def forward(self, x):
        feats = [net(x[:, i:i+1]) for i, net in enumerate(self.feature_nets)]
        return self.output(torch.cat(feats, dim=-1))