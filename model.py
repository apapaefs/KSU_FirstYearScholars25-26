import torch
import torch.nn as nn

#class JetClassifierCNN(nn.Module):
#    """
#    Simple CNN for quark/gluon jet classification.
#    Input:  (batch, 3, 32, 32)  — 3-channel jet images
#    Output: (batch, 1)          — raw logit (apply sigmoid for probability)

#    Uses only Hailo-compatible operations:
#    Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear
#    """
#
#    def __init__(self):
#        super().__init__()
#        self.features = nn.Sequential(
#            # Block 1: 3 -> 16 channels, spatial 32 -> 16
#            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(16),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2),

#            # Block 2: 16 -> 32 channels, spatial 16 -> 8
#            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2),

#            # Block 3: 32 -> 64 channels, spatial 8 -> 4
#            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#        )
#        self.classifier = nn.Sequential(
#            nn.Flatten(),               # 64 * 4 * 4 = 1024
#            nn.Linear(64 * 4 * 4, 128),
#            nn.ReLU(),
#            nn.Linear(128, 1),          # raw logit
#        )
#
#    def forward(self, x):
#        x = self.features(x)
#        x = self.classifier(x)
#        return x

class JetClassifierCNN(nn.Module):
    """
    Simple CNN for quark/gluon jet classification (Hailo-friendly).

    Input:  (batch, 3, 32, 32)  — 3-channel jet images (NCHW)
    Output: (batch, 1)          — raw logit (apply sigmoid for probability)

    Uses only Hailo-compatible operations:
      Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear

    Parameterization:
      c1, c2, c3, c4 control the number of channels in successive conv blocks.
      - If c4 is None, the network has 3 conv blocks (ends at spatial 4×4).
      - If c4 is an int, the network has 4 conv blocks (ends at spatial 2×2).

    IMPORTANT:
      Default values replicate the original network you provided:
        c1=16, c2=32, c3=64, c4=None, fc=128
    """

    def __init__(self, c1=16, c2=32, c3=64, c4=None, fc=128):
        super().__init__()

        layers = []

        # -------------------------
        # Block 1: 3 -> c1 channels
        # Spatial: 32 -> 16 (via MaxPool 2×2)
        # -------------------------
        layers += [
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1),   # preserves 32×32
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 32×32 -> 16×16
        ]

        # -------------------------
        # Block 2: c1 -> c2 channels
        # Spatial: 16 -> 8
        # -------------------------
        layers += [
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),  # preserves 16×16
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16×16 -> 8×8
        ]

        # -------------------------
        # Block 3: c2 -> c3 channels
        # Spatial: 8 -> 4
        # -------------------------
        layers += [
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),  # preserves 8×8
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 8×8 -> 4×4
        ]

        # -------------------------
        # Optional Block 4: c3 -> c4 channels
        # Spatial: 4 -> 2 (only if c4 is provided)
        # -------------------------
        if c4 is not None:
            layers += [
                nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),  # preserves 4×4
                nn.BatchNorm2d(c4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                  # 4×4 -> 2×2
            ]

        self.features = nn.Sequential(*layers)

        # Infer flatten dimension automatically so c1–c4 changes don't break the FC input size.
        # For the default (original) model, this will be 64 * 4 * 4 = 1024.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            n_flat = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, fc),
            nn.ReLU(),
            nn.Linear(fc, 1),  # raw logit
        )

    def forward(self, x):
        # Feature extractor: convolution + pooling blocks
        x = self.features(x)

        # Classifier head: flatten + MLP to a single logit
        x = self.classifier(x)

        return x

def model_one_liner(model, input_shape=(1, 3, 32, 32), device="cpu"):
    """
    Print a one-line summary:
      - total parameters
      - trainable parameters
      - inferred feature-map shape after model.features (if present)
      - inferred flatten dim going into the first Linear (if classifier is present)

    Assumes your model has attributes:
      - model.features (nn.Sequential)
      - model.classifier (nn.Sequential)
    but will degrade gracefully if not.

    Example:
        model = JetClassifierCNN()
        model_one_liner(model)
    """
    model = model.to(device)
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Run a dummy forward through features (if available) to infer shapes
    feat_shape_str = "n/a"
    flat_dim_str = "n/a"

    with torch.no_grad():
        x = torch.zeros(*input_shape, device=device)

        if hasattr(model, "features"):
            feats = model.features(x)
            feat_shape = tuple(feats.shape)  # (batch, C, H, W) typically
            feat_shape_str = str(feat_shape)

            # Flatten dim per example (exclude batch dimension)
            flat_dim = feats[0].numel()
            flat_dim_str = str(flat_dim)

    # One-line print (students can paste into reports)
    print(
        f"{model.__class__.__name__}: "
        f"params={total_params:,} (trainable={trainable_params:,}) | "
        f"features_out={feat_shape_str} | "
        f"flatten_dim={flat_dim_str}"
    )    

if __name__ == "__main__":
    # Quick test: verify shapes
    model = JetClassifierCNN()
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
