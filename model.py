import torch
import torch.nn as nn

class JetClassifierCNN(nn.Module):
    """
    Simple CNN for quark/gluon jet classification.
    Input:  (batch, 3, 32, 32)  — 3-channel jet images
    Output: (batch, 1)          — raw logit (apply sigmoid for probability)

    Uses only Hailo-compatible operations:
    Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 16 channels, spatial 32 -> 16
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16 -> 32 channels, spatial 16 -> 8
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 32 -> 64 channels, spatial 8 -> 4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 64 * 4 * 4 = 1024
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),          # raw logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick test: verify shapes
    model = JetClassifierCNN()
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
