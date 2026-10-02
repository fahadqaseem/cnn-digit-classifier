"""Generates cnn_classifier.ipynb programmatically using nbformat."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"}
}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Homework 3: CNN Digit Classifier

**Course**: AI for Augmented and Virtual Reality (CSC 5991)  
**Author**: Fahad Qaseem Khawar - hr4625  
**Dataset**: MNIST — handwritten digits (0–9)  
**Framework**: PyTorch 2.7.1  
**Device**: Apple MPS (Metal Performance Shaders)  
**GitHub**: [github.com/fahadqaseem/cnn-digit-classifier](https://github.com/fahadqaseem/cnn-digit-classifier)

---

## 🔑 How This Differs from Assignment 2 (Feedforward Network)

| | Assignment 2 — Feedforward | Assignment 3 — CNN (this file) |
|---|---|---|
| **First step** | Flatten image → 784 numbers | Keep image as 2D (1×28×28) |
| **Core layer** | `nn.Linear` (fully connected) | `nn.Conv2d` (convolution) |
| **Spatial awareness** | None — pixels are treated independently | Yes — 3×3 filters slide over the image |
| **Translation invariance** | ❌ | ✅ Same filter fires wherever an edge appears |
| **Pooling** | None | MaxPool2d after each conv layer |
| **Parameters** | ~235K | 421,642 |
| **Test accuracy** | 98.09% | **99.36%** |

> **The key insight:** Assignment 2 destroyed spatial structure immediately by flattening.  
> This CNN *preserves* the 2D structure and learns *where* patterns are — that's why it wins.
"""))

# ── Section 1 ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Imports & Device Setup"))

cells.append(nbf.v4.new_code_cell("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os

# Reproducibility — same seed = same results every run
torch.manual_seed(42)
np.random.seed(42)

# Device selection — prefer MPS (Apple Silicon GPU), then CUDA, then CPU
# DIFFERENCE from A2: same device logic, but MPS matters more here because
# conv layers are much more GPU-friendly than linear layers
device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f'Using device: {device}')
print(f'PyTorch version: {torch.__version__}')
"""))

# ── Section 2 ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Load & Explore MNIST Data

MNIST images are 28×28 grayscale pixels.  
We normalise them using the dataset's known **mean = 0.1307** and **std = 0.3081**.

> **Same as A2** — data loading and preprocessing are identical.  
> The difference starts at the model definition.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Hyperparameters
BATCH_SIZE = 64    # same as A2
EPOCHS     = 10    # same as A2
LR         = 0.001 # same as A2

# Transform: convert to tensor + normalise
# DIFFERENCE from A2: ToTensor() here produces shape [1, 28, 28] (channel, H, W)
# In A2 we immediately flattened this to [784]. Here we keep the 2D structure.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Training samples : {len(train_dataset):,}')
print(f'Test samples     : {len(test_dataset):,}')
print(f'Input shape      : {train_dataset[0][0].shape}')
# → torch.Size([1, 28, 28])  ← this 2D shape is what the CNN uses directly
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualise a grid of sample training images
fig, axes = plt.subplots(3, 8, figsize=(14, 6))
fig.suptitle('Sample MNIST Training Images', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(str(label), fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

# ── Section 3 ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 3. CNN Architecture

```
Input  (1 × 28 × 28)              ← DIFFERENCE: A2 flattened this to 784 immediately
  │
  ├── Conv2d(1→32, 3×3, pad=1)    ← NEW in A3: learns 32 edge/pattern detectors
  │   ReLU
  │   MaxPool2d(2×2)              ← NEW in A3: shrinks 28×28 → 14×14, adds robustness
  │                   → 32 × 14 × 14
  │
  ├── Conv2d(32→64, 3×3, pad=1)   ← NEW in A3: stacks on top of first conv (deeper patterns)
  │   ReLU
  │   MaxPool2d(2×2)              ← NEW in A3: shrinks 14×14 → 7×7
  │                   → 64 × 7 × 7
  │
  ├── Flatten                     ← Only flattened HERE, after spatial features extracted
  │                   → 3136
  │
  ├── Linear(3136 → 128)          ← Similar to A2's hidden layers, but input is richer features
  │   ReLU
  │   Dropout(0.5)
  │
  └── Linear(128 → 10)            ← Same output: 10 class scores
```
"""))

cells.append(nbf.v4.new_code_cell("""\
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ── CONVOLUTIONAL BLOCK 1 ──────────────────────────────────────────────
        # DIFFERENCE from A2: Instead of nn.Linear(784, 256), we use Conv2d.
        # Conv2d slides a 3×3 filter across the image. Each of the 32 filters
        # learns to detect a different pattern (horizontal edges, curves, etc.)
        # padding=1 keeps the output the same spatial size (28×28 stays 28×28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool2d: takes the max in each 2×2 block → halves spatial size
            # 28×28 → 14×14. This makes the network robust to small shifts.
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── CONVOLUTIONAL BLOCK 2 ──────────────────────────────────────────────
        # Second conv layer takes the 32 feature maps from block 1 as input.
        # It learns higher-level patterns (combinations of edges → shapes).
        # 64 filters means 64 different pattern detectors at this level.
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 14×14 → 7×7
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── FULLY CONNECTED HEAD ───────────────────────────────────────────────
        # SIMILARITY to A2: after extracting spatial features, we still use
        # linear layers to classify. The difference is the input is now rich
        # spatial features (3136 values) instead of raw pixels (784 values).
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 64 × 7 × 7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),                 # Dropout rate raised to 0.5 (A2 used 0.2)
            nn.Linear(128, 10)                 # 10 output classes — same as A2
        )

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]  ← 2D image, NOT flattened
        x = self.conv1(x)              # → [batch, 32, 14, 14]
        x = self.conv2(x)              # → [batch, 64, 7, 7]
        x = self.classifier(x)         # → [batch, 10]
        return x


model = DigitCNN().to(device)
print(model)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\\nTotal parameters    : {total:,}')
print(f'Trainable parameters: {trainable:,}')
# Compare: A2 had ~235K parameters, this CNN has 421,642
# More parameters, but more structured → better accuracy
"""))

# ── Section 4 ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Training

- **Loss:** CrossEntropyLoss — same as A2, standard for multi-class problems  
- **Optimiser:** Adam (lr=0.001) — same as A2  
- **Epochs:** 10 — same as A2  

> The training loop is **identical** to A2. Only the model architecture changed.  
> This is intentional — it isolates the effect of switching from FC to CNN.
"""))

cells.append(nbf.v4.new_code_cell("""\
criterion = nn.CrossEntropyLoss()  # same as A2
optimizer = optim.Adam(model.parameters(), lr=LR)  # same as A2

def evaluate(loader):
    \"\"\"Run model on a dataloader, return (avg_loss, accuracy%).\"\"\"
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)
            total_loss  += loss.item() * imgs.size(0)
            preds        = outputs.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += imgs.size(0)
    return total_loss / total, correct / total * 100
"""))

cells.append(nbf.v4.new_code_cell("""\
train_losses, train_accs = [], []
test_losses,  test_accs  = [], []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)          # forward pass through CNN
        loss    = criterion(outputs, labels)
        loss.backward()                # backprop through conv + linear layers
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

    print(f'Epoch {epoch:02d}/{EPOCHS}  |  '
          f'Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  '
          f'Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%')

elapsed = time.time() - start_time
print(f'\\nTraining complete in {elapsed:.1f}s')
print(f'Best test accuracy: {max(test_accs):.2f}%')

os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/cnn_digit_model.pth')
print('Model saved to outputs/cnn_digit_model.pth')
"""))

# ── Section 5 ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Results & Visualisations"))

cells.append(nbf.v4.new_code_cell("""\
# Training curves — compare loss/accuracy across epochs
epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('CNN Training Curves — MNIST (Homework 3, hr4625)', fontsize=13, fontweight='bold')

axes[0].plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2)
axes[0].plot(epochs_range, test_losses,  'r-o', label='Test Loss',  linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss per Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_accs, 'b-o', label='Train Acc', linewidth=2)
axes[1].plot(epochs_range, test_accs,  'r-o', label='Test Acc',  linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy per Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([95, 100])

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Confusion matrix — shows which digits get confused with each other
# Same format as A2 for easy comparison
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs    = imgs.to(device)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Build confusion matrix (pure numpy, no sklearn needed)
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t][p] += 1

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label',      fontsize=12)
ax.set_title('Confusion Matrix — CNN on MNIST (hr4625)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# 25 random test predictions — green = correct, red = wrong
# The CNN should have very few red titles (misclassifications)
indices = np.random.choice(len(test_dataset), 25, replace=False)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle('Sample Predictions — CNN on MNIST (hr4625)', fontsize=14, fontweight='bold')

model.eval()
with torch.no_grad():
    for i, idx in enumerate(indices):
        img, true_label = test_dataset[idx]
        output     = model(img.unsqueeze(0).to(device))
        pred_label = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item() * 100

        ax    = axes[i // 5][i % 5]
        color = 'green' if pred_label == true_label else 'red'
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Pred: {pred_label}\\n({confidence:.1f}%)', color=color, fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Final summary — compare directly with A2
final_loss, final_acc = evaluate(test_loader)
print('=' * 55)
print('FINAL RESULTS — Homework 3 (hr4625)')
print('=' * 55)
print(f'Test Accuracy : {final_acc:.2f}%')
print(f'Test Loss     : {final_loss:.4f}')
print(f'Training Time : {elapsed:.1f}s')
print(f'Device        : {device}')
print()
print('vs. Assignment 2 (Feedforward): 98.09%')
print(f'Improvement   : +{final_acc - 98.09:.2f}%')
print('=' * 55)
"""))

nb.cells = cells

with open('cnn_classifier.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written: cnn_classifier.ipynb")
