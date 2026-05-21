import pandas as pd
import torch
import torch.nn as nn
import math

from client_net import client_network
from new_dataset import data_preparing
from server_net import prediction_net


# ══════════════════════════════════════════════════════════════════
#  راهنمای تفسیر معیارهای ارزیابی
# ══════════════════════════════════════════════════════════════════
METRICS_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║              راهنمای تفسیر معیارهای ارزیابی                    ║
╠══════════════════════════════════════════════════════════════════╣
║  MSE   (Mean Squared Error)        → هر چقدر پایین‌تر، بهتر    ║
║        خطاهای بزرگ را بیشتر جریمه می‌کند                       ║
║                                                                  ║
║  RMSE  (Root Mean Squared Error)   → هر چقدر پایین‌تر، بهتر    ║
║        همان MSE ولی هم‌واحد با مقدار اصلی (مثلاً %)            ║
║                                                                  ║
║  MAE   (Mean Absolute Error)       → هر چقدر پایین‌تر، بهتر    ║
║        به outlier مقاوم‌تر — برای داده‌های ICU مناسب           ║
║                                                                  ║
║  R²    (Coefficient of Determination) → هر چقدر به ۱ نزدیک‌تر ║
║        مثبت: مدل بهتر از میانگین    منفی: مدل ضعیف است         ║
║                                                                  ║
║  MAPE  (Mean Absolute Percentage Error) → هر چقدر پایین‌تر     ║
║        خطا به درصد — برای مقایسه بین target های مختلف          ║
╚══════════════════════════════════════════════════════════════════╝
"""


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


def _compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    محاسبه همه معیارهای ارزیابی روی یک batch یا کل داده.
    preds, targets: هر دو 1D tensor روی CPU.
    """
    preds   = preds.float()
    targets = targets.float()

    mse  = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    mae  = torch.mean(torch.abs(preds - targets))

    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = 1.0 - ss_res / (3 * ss_tot + 1e-8)

    # MAPE: مقادیر نزدیک صفر رو mask می‌کنیم تا تقسیم بر صفر نشه
    nonzero_mask = torch.abs(targets) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = torch.mean(
            torch.abs((targets[nonzero_mask] - preds[nonzero_mask])
                      / targets[nonzero_mask])
        ) * 100.0
    else:
        mape = torch.tensor(float('nan'))

    return {
        "MSE":  mse.item(),
        "RMSE": rmse.item(),
        "MAE":  mae.item(),
        "R2":   r2.item(),
        "MAPE": mape.item(),
    }


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

    # ──────────────────────────────────────────────────────────────
    #  fit
    # ──────────────────────────────────────────────────────────────
    def fit(self, epochs):
        # نمایش راهنمای معیارها فقط یک بار در ابتدا
        print(METRICS_GUIDE)

        history = {
            "loss_train": [],
            "loss_test":  [],
            # train
            "train_MSE":  [], "train_RMSE": [], "train_MAE":  [],
            "train_R2":   [], "train_MAPE": [],
            # test
            "test_MSE":   [], "test_RMSE":  [], "test_MAE":   [],
            "test_R2":    [], "test_MAPE":  [],
        }

        for ep in range(epochs):
            lr_now = lr_at_epoch(ep + 1, self._lr_spec_client)
            self._apply_learning_rate(lr_now)
            self.train_one_epoch()

            tr_metrics = self.evaluate_train()
            te_metrics = self.evaluate_test()

            # ذخیره در history
            history["loss_train"].append(tr_metrics["MSE"])
            history["loss_test"].append(te_metrics["MSE"])
            for key in ("MSE", "RMSE", "MAE", "R2", "MAPE"):
                history[f"train_{key}"].append(tr_metrics[key])
                history[f"test_{key}"].append(te_metrics[key])

            # چاپ خوانا
            self._print_epoch(ep + 1, epochs, tr_metrics, te_metrics)

        return history

    @staticmethod
    def _print_epoch(ep, total_ep, tr, te):
        sep = "─" * 62
        print(f"\n┌{sep}┐")
        print(f"│  Epoch {ep:>3}/{total_ep:<3}{'':>46}│")
        print(f"├{'─'*20}┬{'─'*19}┬{'─'*20}┤")
        print(f"│{'  Metric':^20}│{'  Train':^19}│{'  Test':^20}│")
        print(f"├{'─'*20}┼{'─'*19}┼{'─'*20}┤")
        for name in ("MSE", "RMSE", "MAE", "R2", "MAPE"):
            unit = "%" if name == "MAPE" else ("" if name == "R2" else "")
            tr_v = tr[name]
            te_v = te[name]
            if math.isnan(tr_v) or math.isnan(te_v):
                tr_str = "   N/A"
                te_str = "   N/A"
            else:
                tr_str = f"{tr_v:>12.4f}{unit}"
                te_str = f"{te_v:>12.4f}{unit}"
            print(f"│  {name:<18}│{tr_str:>19}│{te_str:>20}│")
        print(f"└{'─'*20}┴{'─'*19}┴{'─'*20}┘")

    # ──────────────────────────────────────────────────────────────
    #  train
    # ──────────────────────────────────────────────────────────────
    def train_one_epoch(self):
        self.network.train()
        self.prediction.train()
        for x, y, pad_mask in self.data.train_loader:
            x, y, pad_mask = (x.to(self.device),
                               y.to(self.device),
                               pad_mask.to(self.device))
            self.optimizer.zero_grad()
            cls_out   = self.network(x, pad_mask)
            pred      = self.prediction.forward_direct(cls_out)
            task_loss = self.loss_fn(pred, y)
            ae_loss   = self.network.lambda_ae * self.network.last_ae_loss
            (task_loss + ae_loss).backward()
            self.optimizer.step()

    # ──────────────────────────────────────────────────────────────
    #  evaluate
    # ──────────────────────────────────────────────────────────────
    def evaluate_train(self) -> dict:
        self.network.eval()
        self.prediction.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y, pad_mask in self.data.train_loader:
                x, y, pad_mask = (x.to(self.device),
                                   y.to(self.device),
                                   pad_mask.to(self.device))
                cls_out = self.network(x, pad_mask)
                pred    = self.prediction.forward_direct(cls_out)
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return _compute_metrics(all_preds, all_targets)

    def evaluate_test(self) -> dict:
        self.network.eval()
        self.prediction.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y, pad_mask in self.data.test_loader:
                x, y, pad_mask = (x.to(self.device),
                                   y.to(self.device),
                                   pad_mask.to(self.device))
                cls_out = self.network(x, pad_mask)
                pred    = self.prediction.forward_direct(cls_out)
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return _compute_metrics(all_preds, all_targets)

    # ──────────────────────────────────────────────────────────────
    #  knowledge transfer (بدون تغییر)
    # ──────────────────────────────────────────────────────────────
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
