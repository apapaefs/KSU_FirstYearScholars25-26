import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import os
import argparse
import datetime

from load_jets import load_jets, preprocess_all_jets
from model import JetClassifierCNN, model_one_liner

################
# ARGUMENTS    #
################

parser = argparse.ArgumentParser(description="Train quark/gluon jet classifier")
parser.add_argument("--tag", type=str, default="3ch_16-32-64",
                    help="Tag appended to all output filenames (default: 3ch_16-32-64)")
parser.add_argument("--data", type=str, default="data/QG_jets.npz",
                    help="Path to jet data file")
parser.add_argument("--outdir", type=str, default="output",
                    help="Output directory for all results")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--c1", type=int, default=16, help="Conv block 1 channels")
parser.add_argument("--c2", type=int, default=32, help="Conv block 2 channels")
parser.add_argument("--c3", type=int, default=64, help="Conv block 3 channels")
parser.add_argument("--c4", type=int, default=None, help="Conv block 4 channels (optional)")
parser.add_argument("--fc", type=int, default=128, help="FC hidden layer size")
args = parser.parse_args()

TAG = args.tag
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

###########################
# DATA LOADING & CACHING  #
###########################

CACHE_FILE = os.path.join(OUTDIR, f"jet_images_3ch_{TAG}.npz")

X_raw, y = load_jets(args.data)

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

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

################
# TRAINING     #
################

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = JetClassifierCNN(c1=args.c1, c2=args.c2, c3=args.c3,
                          c4=args.c4, fc=args.fc).to(device)

# print model info
model_one_liner(model, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

num_epochs = args.epochs
best_val_loss = float('inf')
train_losses, val_losses, val_accs = [], [], []

model_path = os.path.join(OUTDIR, f"best_jet_classifier_{TAG}.pt")

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
        torch.save(model.state_dict(), model_path)
        print(f"  -> Saved best model (val_loss={val_loss:.4f})")

################
# EVALUATION   #
################

# load best model
model.load_state_dict(torch.load(model_path, map_location=device))
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
test_acc = (test_preds == test_labels).mean()
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
plot_path = os.path.join(OUTDIR, f"training_results_{TAG}.pdf")
plt.savefig(plot_path)
print(f"Saved {plot_path}")
plt.show()

################
# .INFO FILE   #
################

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

info_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.info")
with open(info_path, "w") as f:
    f.write(f"# Jet Classifier Model Info\n")
    f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"tag             = {TAG}\n")
    f.write(f"data            = {args.data}\n")
    f.write(f"architecture    = JetClassifierCNN\n")
    f.write(f"conv_channels   = [{args.c1}, {args.c2}, {args.c3}{',' + str(args.c4) if args.c4 else ''}]\n")
    f.write(f"fc_hidden       = {args.fc}\n")
    f.write(f"input_shape     = (3, 32, 32)\n")
    f.write(f"channels        = [positive_pT, negative_pT, neutral_pT]\n")
    f.write(f"total_params    = {total_params}\n")
    f.write(f"trainable_params= {trainable_params}\n")
    f.write(f"epochs          = {num_epochs}\n")
    f.write(f"batch_size      = {args.batch_size}\n")
    f.write(f"learning_rate   = {args.lr}\n")
    f.write(f"optimizer       = Adam\n")
    f.write(f"loss_function   = BCEWithLogitsLoss\n")
    f.write(f"scheduler       = ReduceLROnPlateau(patience=3, factor=0.5)\n")
    f.write(f"train_samples   = {X_train.shape[0]}\n")
    f.write(f"val_samples     = {X_val.shape[0]}\n")
    f.write(f"test_samples    = {X_test.shape[0]}\n")
    f.write(f"test_accuracy   = {test_acc:.4f}\n")
    f.write(f"test_auc        = {roc_auc:.4f}\n")
    f.write(f"best_val_loss   = {best_val_loss:.4f}\n")
    f.write(f"device          = {device}\n\n")
    f.write(f"# Output files\n")
    f.write(f"model_weights   = best_jet_classifier_{TAG}.pt\n")
    f.write(f"onnx_model      = jet_classifier_{TAG}.onnx\n")
    f.write(f"hailo_alls      = jet_classifier_{TAG}.alls\n")
    f.write(f"hailo_hef       = jet_classifier_{TAG}.hef\n")
    f.write(f"training_plots  = training_results_{TAG}.pdf\n")

print(f"Saved {info_path}")
