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


def _parse_CT_HTTPS_positional(args, chartevents_kw):
    if len(args) > 0 and isinstance(args[0], str):
        chartevents_path = chartevents_kw if chartevents_kw else args[0]
        return (chartevents_path, args[1], args[2], args[3],
                args[4] if len(args) >= 5 else None)
    chartevents_path = chartevents_kw if chartevents_kw else "./CHARTEVENTS.csv"
    return (chartevents_path, args[0], args[1], args[2],
            args[3] if len(args) >= 4 else None)


class CT_HTTPS(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        kw = dict(kwargs)

        chartevents_kw = kw.pop("chartevents_path", None)
        (chartevents_path, w, dataset_name, batch_size, _) = \
            _parse_CT_HTTPS_positional(args, chartevents_kw)

        # پارامترهای داده
        target         = kw.pop("target",         "spO2")
        test_size      = kw.pop("test_size",       0.2)
        normalize_data = kw.pop("normalize_data",  True)

        # پارامترهای client
        lr              = kw.pop("lr",              None)
        client_lr       = kw.pop("client_lr",       0.01)
        d_model         = kw.pop("d_model",         64)
        nhead           = kw.pop("nhead",           4)
        dim_feedforward = kw.pop("dim_feedforward", 128)
        num_cells       = kw.pop("num_cells",       3)
        dropout         = kw.pop("dropout",         0.1)

        # پارامترهای server
        server_lr         = kw.pop("server_lr",         0.01)
        num_primary_caps  = kw.pop("num_primary_caps",  8)
        primary_dim       = kw.pop("primary_dim",       8)
        num_output_caps   = kw.pop("num_output_caps",   4)
        output_dim        = kw.pop("output_dim",        16)
        num_routing       = kw.pop("num_routing",       3)
        server_fc_hidden1 = kw.pop("server_fc_hidden1", 128)
        server_fc_hidden2 = kw.pop("server_fc_hidden2", 64)

        # سازگاری با lr قدیمی
        if lr is not None:
            client_lr = lr
            server_lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initial_client_lr = lr_at_epoch(1, client_lr)
        initial_server_lr = lr_at_epoch(1, server_lr)
        self._lr_spec_client = client_lr
        self._lr_spec_server = server_lr

        n_features = 4 if dataset_name == "metavision" else 5

        # ساخت client network
        self.network = client_network(
            w=w,
            n_features_input=n_features,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_cells=num_cells,
            dropout=dropout,
            lr=initial_client_lr,
        ).to(self.device)

        # FIX اصلی: بُعد CLS که client میده d_model * n_features هست
        # چون در client_network هر feature یه latent d_model داره و کنار هم concat میشن
        d_model_server = d_model * n_features

        # ساخت server (prediction) network
        self.prediction = prediction_net(
            d_model_server=d_model_server,
            lr=initial_server_lr,
            device=self.device,
            num_primary_caps=num_primary_caps,
            primary_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            num_routing=num_routing,
            fc_hidden1=server_fc_hidden1,
            fc_hidden2=server_fc_hidden2,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.prediction.parameters()),
            lr=initial_client_lr,
        )

        df = pd.read_csv(chartevents_path, low_memory=False)
        self.data = data_preparing(
            df, dataset_name, w,
            test_size=test_size,
            target=target,
            batch_size=batch_size,
            normalize=normalize_data,
        )

        self.loss_fn = nn.MSELoss()
        self.L1Loss  = nn.L1Loss()

    def _apply_learning_rate(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def fit(self, epochs):
        history = {"loss_train": [], "loss_test": []}
        for ep in range(epochs):
            lr_now = lr_at_epoch(ep + 1, self._lr_spec_client)
            self._apply_learning_rate(lr_now)
            self.train_one_epoch()
            tr, te = self.evaluate_one_epoch()
            print(f"epoch {ep+1}/{epochs}  train={tr:.4f}  test={te:.4f}")
            history["loss_train"].append(tr.item())
            history["loss_test"].append(te.item())
        return history

    def train_one_epoch(self):
        self.network.train()
        self.prediction.train()
        for x, y, pad_mask in self.data.train_loader:
            x, y, pad_mask = x.to(self.device), y.to(self.device), pad_mask.to(self.device)
            self.optimizer.zero_grad()
            cls_out   = self.network(x, pad_mask)
            pred      = self.prediction.forward_direct(cls_out)
            task_loss = self.loss_fn(pred, y)
            ae_loss   = self.network.lambda_ae * self.network.last_ae_loss
            (task_loss + ae_loss).backward()
            self.optimizer.step()

    def evaluate_one_epoch(self):
        self.network.eval()
        self.prediction.eval()
        with torch.no_grad():
            tr = self._eval_loader(self.data.train_loader)
            te = self._eval_loader(self.data.test_loader)
        return tr, te

    def _eval_loader(self, loader):
        preds, labels = [], []
        for x, y, pad_mask in loader:
            x, y, pad_mask = x.to(self.device), y.to(self.device), pad_mask.to(self.device)
            with torch.no_grad():
                cls_out = self.network(x, pad_mask)
                pred    = self.prediction.forward_direct(cls_out)
            preds.append(pred.cpu())
            labels.append(y.cpu())
        
        preds  = torch.cat(preds)
        labels = torch.cat(labels)
        
        mse  = ((preds - labels)**2).mean()
        mae  = (preds - labels).abs().mean()
        rmse = mse.sqrt()
        ss_res = ((labels - preds)**2).sum()
        ss_tot = ((labels - labels.mean())**2).sum()
        r2   = 1 - ss_res / ss_tot
        
        return {"MSE": mse.item(), "MAE": mae.item(), "RMSE": rmse.item(), "R²": r2.item()}
    def get_knowledge(self, CT_object):
        source_aes = CT_object.network.feature_aes
        n_shared   = min(self.network.n_features, len(source_aes))
        for i in range(n_shared):
            losses = [self.compute_autoencoder_loss(ae, i) for ae in source_aes]
            idx    = torch.argmin(torch.stack(losses))
            print(f"feature {i} chooses AE {idx.item()}")
            self.network.feature_aes[i].load_state_dict(source_aes[idx].state_dict())

    def compute_autoencoder_loss(self, autoencoder, feature_idx):
        autoencoder = autoencoder.to(self.device)
        autoencoder.eval()
        total, n = 0, 0
        with torch.no_grad():
            for x, _, pad_mask in self.data.train_loader:
                x, pad_mask = x.to(self.device), pad_mask.to(self.device)
                xi       = x[:, :, feature_idx:feature_idx+1]
                _, x_hat = autoencoder(xi)
                loss     = self.L1Loss(
                    x_hat * pad_mask.unsqueeze(-1),
                    xi    * pad_mask.unsqueeze(-1)
                )
                total += x.shape[0] * loss
                n     += x.shape[0]
        return total / n
