"""MLS-Bench starter design for the editable inverse-folding region."""


class MPNNEncoderLayer(nn.Module):
    """Message-passing layer for a protein residue graph."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_node = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask):
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        D = int(h_V.shape[-1])
        h_V_neighbors = torch.gather(
            h_V.unsqueeze(2).expand(-1, -1, K, -1),
            1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )
        h_V_expand = h_V.unsqueeze(2).expand_as(h_V_neighbors)
        messages = self.W_msg(torch.cat([h_V_expand, h_V_neighbors, h_E], dim=-1))
        mask_attend = torch.gather(
            mask.unsqueeze(2).expand(-1, -1, K),
            1,
            E_idx.clamp(0, L - 1),
        ).unsqueeze(-1)
        messages = messages * mask_attend
        aggregate = messages.sum(dim=2) / mask_attend.sum(dim=2).clamp(min=1)
        h_V = self.norm1(h_V + self.dropout(aggregate))
        update = self.W_node(torch.cat([h_V, aggregate], dim=-1))
        h_V = self.norm2(h_V + self.dropout(update))
        return h_V * mask.unsqueeze(-1)


class StructureEncoder(nn.Module):
    """Encode backbone coordinates as per-residue hidden representations."""

    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        k_neighbors=30,
        dropout=0.1,
        num_rbf=16,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.node_embed = nn.Linear(12, hidden_dim)
        self.edge_embed = nn.Linear(num_rbf + 3, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MPNNEncoderLayer(hidden_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, X, mask):
        X_ca = X[:, :, 1, :]
        E_idx, distances = knn_graph(X_ca, mask, self.k_neighbors)
        K = E_idx.shape[2]
        node_features = torch.cat([_dihedrals(X), _orientations(X)], dim=-1)
        rbf = _rbf(distances, device=X.device)
        neighbors = torch.gather(
            X_ca.unsqueeze(2).expand(-1, -1, K, -1),
            1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        directions = F.normalize(neighbors - X_ca.unsqueeze(2), dim=-1)
        h_V = self.node_embed(node_features)
        h_E = self.edge_embed(torch.cat([rbf, directions], dim=-1))
        for layer in self.layers:
            h_V = layer(h_V, h_E, E_idx, mask)
        return h_V


class InverseFoldingModel(nn.Module):
    """Predict one amino-acid distribution for every valid residue."""

    def __init__(
        self,
        hidden_dim=128,
        num_encoder_layers=3,
        k_neighbors=30,
        dropout=0.1,
        num_rbf=16,
    ):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        logits = self.decoder(self.encoder(X, mask))
        return F.log_softmax(logits, dim=-1)


CONFIG_OVERRIDES = {}

