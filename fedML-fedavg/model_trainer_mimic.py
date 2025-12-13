import os
import time
import logging
import torch

from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS


def _ensure_logger(args):
    logger = logging.getLogger("mimic_train")

    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path = getattr(args, "log_path", None)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._configured = True
    return logger


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def _multilabel_metrics(logits, targets, thresh=0.5, eps=1e-12):
    probs = _sigmoid(logits)
    preds = (probs >= thresh).to(torch.int32)
    t = (targets >= 0.5).to(torch.int32)

    tp = (preds & t).sum().item()
    fp = (preds & (1 - t)).sum().item()
    fn = ((1 - preds) & t).sum().item()

    micro_prec = tp / (tp + fp + eps)
    micro_rec = tp / (tp + fn + eps)
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + eps)

    per_label_correct = (preds == t).to(torch.float32).mean(dim=0)  # [C]
    mean_label_acc = per_label_correct.mean().item()

    return micro_f1, mean_label_acc


class ModelTrainerMIMIC(ModelTrainerCLS):
    def __init__(self, model, args):
        super().__init__(model, args)
        self._logger = _ensure_logger(args)

        header = getattr(args, "log_header_printed", False)
        if not header:
            self._logger.info("===== MIMIC-CXR ResNet-50 FedAvg =====")
            self._logger.info(f"Args: epochs={getattr(args,'epochs',None)} "
                              f"batch_size={getattr(args,'batch_size',None)} "
                              f"lr={getattr(args,'learning_rate',getattr(args,'lr',None))} "
                              f"weight_decay={getattr(args,'weight_decay',None)} "
                              f"seed={getattr(args,'seed',None)} "
                              f"log_path={getattr(args,'log_path',None)}")
            setattr(args, "log_header_printed", True)

    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        lr = float(getattr(args, "learning_rate", getattr(args, "lr", 1e-4)))
        wd = float(getattr(args, "weight_decay", 0.0))
        mom = float(getattr(args, "momentum", 0.9))
        opt_name = str(getattr(args, "client_optimizer", "adam")).lower()
        epochs = int(getattr(args, "epochs", 1))

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        criterion = torch.nn.BCEWithLogitsLoss()

        round_idx = int(getattr(args, "round_idx", 0))
        global_epochs = int(getattr(args, "comm_round", 1)) * epochs
        base_epoch = round_idx * epochs

        for local_ep in range(epochs):
            ep_start = time.time()
            ep_loss = 0.0
            n = 0

            for x, y in train_data:
                x = x.to(device)
                y = y.to(device).float()

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                ep_loss += loss.item() * bs
                n += bs

            train_loss = ep_loss / max(1, n)

            epoch_num = base_epoch + local_ep + 1
            self._logger.info(
                f"Epoch {epoch_num:03d}/{global_epochs:03d} | Train Loss {train_loss:.4f} | "
                f"Val Loss N/A | Val Acc N/A"
            )

    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()

        criterion = torch.nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_n = 0

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_data:
                x = x.to(device)
                y = y.to(device).float()

                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_n += bs

                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())

        val_loss = total_loss / max(1, total_n)

        if len(all_logits) > 0:
            logits_cat = torch.cat(all_logits, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            micro_f1, mean_label_acc = _multilabel_metrics(logits_cat, targets_cat)
            val_acc_pct = mean_label_acc * 100.0
        else:
            micro_f1, val_acc_pct = 0.0, 0.0

        round_idx = int(getattr(args, "round_idx", 0))
        epochs = int(getattr(args, "epochs", 1))
        global_epochs = int(getattr(args, "comm_round", 1)) * epochs
        epoch_num = min((round_idx + 1) * epochs, global_epochs)

        self._logger.info(
            f"Epoch {epoch_num:03d}/{global_epochs:03d} | Train Loss N/A | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc_pct:.2f}% | microF1 {micro_f1:.4f}"
        )

        test_total = int(total_n)
        test_correct = int(round((val_acc_pct / 100.0) * test_total))

        return {
            "test_loss": float(val_loss),
            "test_total": test_total,
            "test_correct": test_correct,

            # Extra metrics (safe to include)
            "val_acc": float(val_acc_pct),
            "micro_f1": float(micro_f1),
        }
