"""Fixed MLS-Bench helpers used only by the real GPU contract smoke."""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_AA = 20


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)


def _dihedrals(X, eps=1e-7):
    X_flat = X[:, :, :3, :].reshape(int(X.shape[0]), -1, 3)
    U = F.normalize(X_flat[:, 1:, :] - X_flat[:, :-1, :], dim=-1)
    u_2, u_1, u_0 = U[:, :-2, :], U[:, 1:-1, :], U[:, 2:, :]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cos_d = (n_2 * n_1).sum(-1).clamp(-1 + eps, 1 - eps)
    sin_d = (torch.cross(n_2, n_1, dim=-1) * u_1).sum(-1)
    D = torch.stack([cos_d, sin_d.clamp(-1 + eps, 1 - eps)], dim=-1)
    D = F.pad(D, (0, 0, 1, 2))
    return D.reshape(int(X.shape[0]), -1, 6)[:, : int(X.shape[1]), :]


def _orientations(X):
    fwd = F.normalize(X[:, 1:, 1, :] - X[:, :-1, 1, :], dim=-1)
    fwd = F.pad(fwd, (0, 0, 0, 1))
    u = F.normalize(X[:, :, 2, :] - X[:, :, 1, :], dim=-1)
    b = F.normalize(fwd - (fwd * u).sum(-1, keepdim=True) * u, dim=-1)
    return torch.cat([fwd, b], dim=-1)


def knn_graph(X_ca, mask, k=30):
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    delta = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)
    distances = mask_2d * torch.sqrt((delta**2).sum(-1) + 1e-6)
    distances = distances + (1 - mask_2d) * 1e6
    return tuple(
        reversed(
            torch.topk(
                distances,
                min(k, int(distances.shape[-1])),
                dim=-1,
                largest=False,
            )
        )
    )
