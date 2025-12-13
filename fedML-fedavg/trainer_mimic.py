# trainer_mimic.py

import torch
import torch.nn as nn
from fedml.core import ClientTrainer


class MIMICTrainer(ClientTrainer):
    """FedML ClientTrainer wrapper for MIMIC-CXR + ResNet-50."""

    def get_model_params(self):
        # FedML expects a plain state_dict for aggregation
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        """Train on one client partition for args.epochs (local epochs)."""
        self.model.to(device)
        self.model.train()

        # Read LR from args; fall back to 1e-4 if missing
        lr = getattr(args, "learning_rate", 1e-4)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        local_epochs = getattr(args, "epochs", 1)

        for _ in range(local_epochs):
            for images, labels in train_data:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    def test(self, test_data, device, args):
        self.model.to(device)
        self.model.eval()

        criterion = nn.BCEWithLogitsLoss().to(device)
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for images, labels in test_data:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).float().sum().item()
                total += labels.numel()

        avg_loss = total_loss / max(len(test_data), 1)
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc
