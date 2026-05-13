import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class ViTFineTuner:
    """Fine-tunes a pretrained ViT model on K-line chart images."""

    def __init__(self, config: dict) -> None:
        self.config = config["model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.model.to(self.device)

    def _build_model(self) -> nn.Module:
        model = timm.create_model(
            self.config["name"],
            pretrained=self.config.get("pretrained", True),
            num_classes=self.config.get("num_classes", 2),
        )
        # Freeze first N transformer blocks
        n_freeze = self.config.get("freeze_layers", 8)
        for i, block in enumerate(model.blocks):
            if i < n_freeze:
                for param in block.parameters():
                    param.requires_grad = False
        return model

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        lr = self.config.get("learning_rate", 1e-4)
        wd = self.config.get("weight_decay", 1e-2)
        max_epochs = self.config.get("max_epochs", 20)
        patience = self.config.get("early_stop_patience", 5)
        save_path = Path(self.config.get("model_save_path", "models/best_model.pth"))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Device:          {self.device}")
        print(f"Model:           {self.config['name']}")
        print(f"Trainable params:{n_params:,}")
        print(f"Train samples:   {n_train}  ({len(train_loader)} batches)")
        print(f"Val samples:     {n_val}  ({len(val_loader)} batches)")
        print(f"Max epochs:      {max_epochs}  (early stop patience={patience})")
        print(f"LR:              {lr}  weight_decay={wd}")
        print(f"Save path:       {save_path}")
        print("-" * 60)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=wd,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        no_improve = 0
        history = []

        for epoch in range(max_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self._eval_epoch(val_loader, criterion)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            improved = val_loss < best_val_loss
            marker = " *" if improved else f"  (no improve {no_improve + 1}/{patience})"
            print(
                f"Epoch {epoch+1:>3}/{max_epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_acc={val_acc:.4f}"
                f"  lr={lr_now:.2e}"
                f"{marker}"
            )

            if improved:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        print("-" * 60)
        print(f"Best val_loss: {best_val_loss:.4f}  model saved → {save_path}")

        # Save training history
        history_path = save_path.parent / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return history

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate on test set and return metrics dict."""
        save_path = Path(self.config.get("model_save_path", "models/best_model.pth"))
        if save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

        self._save_confusion_matrix(metrics["confusion_matrix"], save_path.parent)
        return metrics

    def _train_epoch(self, loader: DataLoader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _eval_epoch(self, loader: DataLoader, criterion) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        avg_loss = total_loss / max(len(loader), 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    def _save_confusion_matrix(self, cm: list, output_dir: Path) -> None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Hold", "Buy"])
        ax.set_yticklabels(["Hold", "Buy"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(str(output_dir / "confusion_matrix.png"))
        plt.close(fig)
