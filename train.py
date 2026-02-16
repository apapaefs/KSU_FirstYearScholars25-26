import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import os

from load_jets import load_jets, preprocess_all_jets
from model import JetClassifierCNN, model_one_liner

###########################
# DATA LOADING & CACHING  #
###########################

CACHE_FILE = "jet_images_3ch.npz"

jetpath = "QG_jets.npz"
X_raw, y = load_jets(jetpath)

# preprocess or load from cache
if os.path.exists(CACHE_FILE):
    print(f"Loading cached images from {CACHE_FILE} ...")
    cached = np.load(CACHE_FILE)
    images = cached["images"]
    y = cached["y"]
    print(f"  Loaded: images {images.shape}, labels {y.shape}")
else:
    print("Converting jets to 3-channel images (this may take a few minutes)...")
    images = preprocess_all_jets(X_raw, R=0.4, npixels=32, pt_min=0.0, normalize=True)
    np.savez_compressed(CACHE_FILE, images=images, y=y)
    print(f"  Saved cache to {CACHE_FILE}")

################
# DATA SPLIT   #
################

# 50k train, 25k val, 25k test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    images, y, test_size=0.5, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# wrap in PyTorch datasets
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val).float())
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test).float())

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

################
# TRAINING     #
################

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = JetClassifierCNN().to(device)

# print model info
model_one_liner(model, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

num_epochs = 30
best_val_loss = float('inf')
train_losses, val_losses, val_accs = [], [], []

for epoch in range(num_epochs):
    # --- train ---
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x).squeeze(-1)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    train_loss = running_loss / len(train_ds)

    # --- validate ---
    model.eval()
    val_loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            val_loss_sum += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_loss = val_loss_sum / len(val_ds)
    val_acc = correct / total
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:2d}/{num_epochs}  "
          f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
          f"Val Acc: {val_acc:.4f}  LR: {lr:.1e}")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_jet_classifier.pt")
        print(f"  -> Saved best model (val_loss={val_loss:.4f})")

################
# EVALUATION   #
################

# load best model
model.load_state_dict(torch.load("best_jet_classifier.pt", map_location=device))
model.eval()

# collect test predictions
test_probs, test_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x).squeeze(-1)
        probs = torch.sigmoid(logits)
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(batch_y.numpy())

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)

# ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

# classification report
test_preds = (test_probs > 0.5).astype(int)
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"AUC: {roc_auc:.4f}")
print(classification_report(test_labels, test_preds, target_names=["Gluon", "Quark"]))

################
# PLOTS        #
################

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Loss curves
axes[0].plot(range(1, num_epochs+1), train_losses, label="Train")
axes[0].plot(range(1, num_epochs+1), val_losses, label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend()
axes[0].set_title("Training and Validation Loss")

# Accuracy curve
axes[1].plot(range(1, num_epochs+1), val_accs)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")

# ROC curve
axes[2].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[2].set_xlabel("False Positive Rate (Gluon Efficiency)")
axes[2].set_ylabel("True Positive Rate (Quark Efficiency)")
axes[2].legend()
axes[2].set_title("ROC Curve")

plt.tight_layout()
plt.savefig("training_results.pdf")
print("Saved training_results.pdf")
plt.show()
