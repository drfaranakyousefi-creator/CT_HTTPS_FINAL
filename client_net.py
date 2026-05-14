import torch
import torch.nn as nn
import torch.optim as optim
import math


# ==========================================================
# Positional Encoding
# ==========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==========================================================
# Per-feature AutoEncoder
# 1 -> hidden -> latent -> hidden -> 1
# ==========================================================
class FeatureAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        # x : (B,W,1)

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return z, x_hat


# ==========================================================
# Transformer Cell
# ==========================================================
class TransformerAutoEncoderCell(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):

        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

        return x


# ==========================================================
# Client Network
# ==========================================================
class client_network(nn.Module):
    def __init__(
        self,
        w,
        n_features_input,
        d_model=8,          # same as latent_dim
        nhead=4,
        dim_feedforward=64,
        num_cells=3,
        dropout=0.1,
        lr=0.01,
        lambda_ae=0.1
    ):
        super().__init__()

        self.w = w
        self.n_features = n_features_input
        self.d_model = d_model
        self.lambda_ae = lambda_ae

        # -----------------------------------
        # Per feature AutoEncoders
        # -----------------------------------
        self.feature_aes = nn.ModuleList([
            FeatureAutoEncoder(d_model)
            for _ in range(n_features_input)
        ])

        # reconstruction loss
        self.mse = nn.MSELoss()

        # -----------------------------------
        # CLS
        # -----------------------------------
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model * n_features_input)
        )

        # -----------------------------------
        # Positional
        # -----------------------------------
        self.pos_encoding = PositionalEncoding(
            d_model * n_features_input,
            max_len=w + 2,
            dropout=dropout
        )

        # -----------------------------------
        # Transformer cells
        # -----------------------------------
        self.cells = nn.ModuleList([
            TransformerAutoEncoderCell(
                d_model * n_features_input,
                nhead,
                dim_feedforward,
                dropout
            )
            for _ in range(num_cells)
        ])

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr
        )

        self.last_ae_loss = None


    @staticmethod
    def _prepend_cls_key_padding(pad_mask):

        kpm = torch.zeros(
            pad_mask.size(0),
            pad_mask.size(1) + 1,
            dtype=torch.bool,
            device=pad_mask.device
        )

        kpm[:, 1:] = pad_mask < 0.5

        return kpm


    @staticmethod
    def _zero_pad_timesteps(x, pad_mask):

        cls_h = x[:, :1, :]
        ts = x[:, 1:, :] * pad_mask.unsqueeze(-1)

        return torch.cat([cls_h, ts], dim=1)


    def forward(self, x, pad_mask=None):

        # x: (B,W,N)

        B = x.size(0)

        if pad_mask is None:
            pad_mask = torch.ones(
                B,
                self.w,
                device=x.device
            )

        z_list = []
        ae_loss = 0.0

        # =====================================
        # per-feature AE
        # =====================================
        for i in range(self.n_features):

            xi = x[:, :, i:i+1]

            z, x_hat = self.feature_aes[i](xi)

            z_list.append(z)

            ae_loss += self.mse(
                x_hat * pad_mask.unsqueeze(-1),
                xi * pad_mask.unsqueeze(-1)
            )

        self.last_ae_loss = ae_loss / self.n_features

        # concat
        x = torch.cat(z_list, dim=-1)

        # =====================================
        # CLS
        # =====================================
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # =====================================
        # positional
        # =====================================
        x = self.pos_encoding(x)
        x = self._zero_pad_timesteps(x, pad_mask)

        key_padding_mask = self._prepend_cls_key_padding(
            pad_mask
        )

        # =====================================
        # transformer
        # =====================================
        for cell in self.cells:
            x = cell(
                x,
                key_padding_mask=key_padding_mask
            )
            x = self._zero_pad_timesteps(
                x,
                pad_mask
            )

        cls_out = x[:, 0, :]

        return cls_out


    def train_one_batch(self, cls_out, grad):

        self.optimizer.zero_grad()

        # main loss
        cls_out.backward(
            grad,
            retain_graph=True
        )

        # AE loss
        ae_loss = self.lambda_ae * self.last_ae_loss
        ae_loss.backward()

        self.optimizer.step()

        return ae_loss.item()