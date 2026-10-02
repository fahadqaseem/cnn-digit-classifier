"""
CNN Digit Classifier — Assignment 3
Trains a Convolutional Neural Network on MNIST.
Saves training curves, confusion matrix, and sample predictions to outputs/.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import time
import os

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── device ───────────────────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 0.001

# ── data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean / std
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples : {len(train_dataset):,}")
print(f"Test samples     : {len(test_dataset):,}")

# ── model ─────────────────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    """
    CNN Architecture:
      Input  : 1 × 28 × 28
      Conv1  : 32 filters, 3×3, ReLU → MaxPool 2×2  →  32 × 13 × 13
      Conv2  : 64 filters, 3×3, ReLU → MaxPool 2×2  →  64 ×  6 ×  6
      Flatten:  64 × 6 × 6 = 2304
      FC1    : 2304 → 128, ReLU, Dropout(0.5)
      FC2    : 128  → 10  (logits)
    """
    def __init__(self):
        super().__init__()

        # ── convolutional block 1 ──────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 28×28 → 14×14
        )

        # ── convolutional block 2 ──────────────────────────────────────────
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 14×14 → 7×7
        )

        # ── classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),             # 3136 → 128
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x


model = DigitCNN().to(device)
print(f"\nModel architecture:\n{model}\n")

# count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable:,}")

# ── loss & optimiser ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── training loop ─────────────────────────────────────────────────────────────
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    return total_loss / total, correct / total * 100


print("\n" + "="*60)
print("Training")
print("="*60)

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    tr_loss = running_loss / total
    tr_acc  = correct / total * 100
    te_loss, te_acc = evaluate(test_loader)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS}  |  "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  "
          f"Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%")

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed:.1f}s")
print(f"Best test accuracy: {max(test_accs):.2f}%")

# ── save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "outputs/cnn_digit_model.pth")
print("Model saved to outputs/cnn_digit_model.pth")

# ── plot 1: training curves ───────────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("CNN Training Curves — MNIST", fontsize=14, fontweight="bold")

# loss
axes[0].plot(epochs_range, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=5)
axes[0].plot(epochs_range, test_losses,  "r-o", label="Test Loss",  linewidth=2, markersize=5)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss per Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# accuracy
axes[1].plot(epochs_range, train_accs, "b-o", label="Train Acc", linewidth=2, markersize=5)
axes[1].plot(epochs_range, test_accs,  "r-o", label="Test Acc",  linewidth=2, markersize=5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Accuracy per Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([95, 100])

plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/training_curves.png")

# ── plot 2: confusion matrix ──────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Build confusion matrix manually (avoids sklearn/scipy dependency)
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t][p] += 1

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label",      fontsize=12)
ax.set_title("Confusion Matrix — CNN on MNIST Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/confusion_matrix.png")

# ── plot 3: sample predictions ────────────────────────────────────────────────
# pick 25 random test images
test_dataset_raw = datasets.MNIST(root="./data", train=False, download=False,
                                   transform=transform)
indices = np.random.choice(len(test_dataset_raw), 25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Sample Predictions — CNN on MNIST", fontsize=14, fontweight="bold")

model.eval()
with torch.no_grad():
    for i, idx in enumerate(indices):
        img, true_label = test_dataset_raw[idx]
        img_tensor = img.unsqueeze(0).to(device)
        output     = model(img_tensor)
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item() * 100

        ax = axes[i // 5][i % 5]
        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Pred: {pred_label}\n({confidence:.1f}%)", color=color, fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/sample_predictions.png")

# ── final summary ─────────────────────────────────────────────────────────────
final_test_loss, final_test_acc = evaluate(test_loader)
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Test Accuracy : {final_test_acc:.2f}%")
print(f"Test Loss     : {final_test_loss:.4f}")
print(f"Training Time : {elapsed:.1f}s")
print(f"Device        : {device}")
print("="*60)
