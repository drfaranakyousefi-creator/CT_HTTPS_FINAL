import pandas as pd
import torch
import torch.nn as nn

from client_net import client_network
from new_dataset import data_preparing
from server_net import prediction_net


def lr_at_epoch(epoch_1based, lr_spec):

    if lr_spec is None:
        raise ValueError("lr cannot be None")

    if not isinstance(lr_spec, dict):
        return float(lr_spec)

    milestones = sorted(lr_spec.keys())

    for m in milestones:
        if epoch_1based <= int(m):
            return float(lr_spec[m])

    return float(lr_spec[milestones[-1]])


def _initial_scalar_lr(lr, override):

    if override is not None:
        return float(override)

    return lr_at_epoch(1, lr)


def _parse_CT_HTTPS_positional(args, chartevents_kw):

    if isinstance(args[0], str):

        chartevents_path = (
            chartevents_kw
            if chartevents_kw
            else args[0]
        )

        return (
            chartevents_path,
            args[1],
            args[2],
            args[3],
            args[4] if len(args) >= 5 else None
        )

    chartevents_path = (
        chartevents_kw
        if chartevents_kw
        else "./CHARTEVENTS.csv"
    )

    return (
        chartevents_path,
        args[0],
        args[1],
        args[2],
        args[3] if len(args) >= 4 else None
    )


class CT_HTTPS(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        kw = dict(kwargs)

        chartevents_kw = kw.pop(
            "chartevents_path",
            None
        )

        (
            chartevents_path,
            w,
            dataset_name,
            batch_size,
            _
        ) = _parse_CT_HTTPS_positional(
            args,
            chartevents_kw
        )

        target = kw.pop("target", "spO2")
        lr = kw.pop("lr", 0.01)

        d_model = kw.pop("d_model", 8)
        nhead = kw.pop("nhead", 4)
        dim_feedforward = kw.pop(
            "dim_feedforward",
            64
        )

        num_cells = kw.pop("num_cells", 3)
        dropout = kw.pop("dropout", 0.1)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._lr_spec = lr
        initial_lr = lr_at_epoch(1, lr)   # FIX: scalar برای Adam

        if dataset_name == "metavision":
            n_features = 4
        else:
            n_features = 5

        self.network = client_network(
            w=w,
            n_features_input=n_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_cells=num_cells,
            dropout=dropout,
            lr=initial_lr           # FIX: scalar نه dict
        ).to(self.device)

        self.prediction = prediction_net(
            d_model=d_model,
            lr=initial_lr,          # FIX: scalar نه dict
            device=self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters())
            +
            list(self.prediction.parameters()),
            lr=initial_lr           # FIX: scalar نه dict
        )

        df = pd.read_csv(
            chartevents_path
        )

        self.data = data_preparing(
            df,
            dataset_name,
            w,
            target=target,
            batch_size=batch_size
        )

        self.loss_fn = nn.MSELoss()
        self.L1Loss = nn.L1Loss()


    def _apply_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr


    def fit(self, epochs):

        history = {
            "loss_train": [],
            "loss_test": []
        }

        for ep in range(epochs):

            lr_now = lr_at_epoch(
                ep + 1,
                self._lr_spec
            )

            self._apply_learning_rate(
                lr_now
            )

            self.train_one_epoch()

            tr, te = (
                self.evaluate_one_epoch()
            )

            print(
                f"epoch {ep+1}/{epochs} "
                f"train={tr:.4f} "
                f"test={te:.4f}"
            )

            history["loss_train"].append(
                tr.item()
            )

            history["loss_test"].append(
                te.item()
            )

        return history


    def train_one_epoch(self):

        self.network.train()
        self.prediction.train()

        for x, y, pad_mask in self.data.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)
            pad_mask = pad_mask.to(
                self.device
            )

            self.optimizer.zero_grad()

            cls_out = self.network(
                x,
                pad_mask
            )

            pred = self.prediction.forward_direct(
                cls_out
            )

            task_loss = self.loss_fn(
                pred,
                y
            )

            ae_loss = (
                self.network.lambda_ae
                *
                self.network.last_ae_loss
            )

            total_loss = (
                task_loss + ae_loss
            )

            total_loss.backward()

            self.optimizer.step()


    def evaluate_one_epoch(self):

        self.network.eval()
        self.prediction.eval()

        with torch.no_grad():

            train_loss = self._eval_loader(
                self.data.train_loader
            )

            test_loss = self._eval_loader(
                self.data.test_loader
            )

        return train_loss, test_loss


    def _eval_loader(self, loader):

        total = 0
        n = 0

        for x, y, pad_mask in loader:

            x = x.to(self.device)
            y = y.to(self.device)
            pad_mask = pad_mask.to(
                self.device
            )

            cls_out = self.network(
                x,
                pad_mask
            )

            pred = self.prediction.forward_direct(
                cls_out
            )

            total += (
                x.shape[0]
                *
                self.loss_fn(pred, y)
            )

            n += x.shape[0]

        return total / n


    def get_knowledge(self, CT_object):

        source_aes = (
            CT_object.network.feature_aes
        )

        n_shared = min(
            self.network.n_features,
            len(source_aes)
        )

        for i in range(n_shared):

            losses = []

            for ae in source_aes:

                l = self.compute_autoencoder_loss(
                    ae,
                    i
                )

                losses.append(l)

            idx = torch.argmin(
                torch.stack(losses)
            )

            print(
                f"feature {i} "
                f"chooses AE {idx}"
            )

            self.network.feature_aes[i].load_state_dict(
                source_aes[idx].state_dict()
            )


    def compute_autoencoder_loss(
        self,
        autoencoder,
        feature_idx
    ):

        autoencoder = autoencoder.to(
            self.device
        )

        autoencoder.eval()

        total = 0
        n = 0

        with torch.no_grad():

            for x, _, pad_mask in self.data.train_loader:

                x = x.to(self.device)
                pad_mask = pad_mask.to(
                    self.device
                )

                xi = x[
                    :,
                    :,
                    feature_idx:feature_idx+1
                ]

                _, x_hat = autoencoder(
                    xi
                )

                loss = self.L1Loss(
                    x_hat *
                    pad_mask.unsqueeze(-1),

                    xi *
                    pad_mask.unsqueeze(-1)
                )

                total += (
                    x.shape[0]
                    *
                    loss
                )

                n += x.shape[0]

        return total / n
