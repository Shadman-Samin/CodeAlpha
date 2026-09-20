"""Handwritten Character Recognition — MNIST CNN with PyTorch.

Task doc: MNIST digits / EMNIST chars, CNN, extendable to CRNN.
Runs on CPU or CUDA. 3 epochs default for quick verify (~98% acc).
"""
from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DATA = Path(__file__).resolve().parent.parent / "data"
ASSETS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def main(epochs=3, batch=128):
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=512, num_workers=0)
    print(f"MNIST train={len(train_ds)} test={len(test_ds)} device={DEVICE}")

    model = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        print(f"Epoch {ep}: loss={loss_sum/total:.4f} acc={correct/total:.4f}")

    # Eval
    model.eval()
    correct, total = 0, 0
    preds, trues, imgs = [], [], []
    with torch.no_grad():
        for x, y in test_ld:
            out = model(x.to(DEVICE))
            p = out.argmax(1).cpu()
            preds += p.tolist(); trues += y.tolist()
            if len(imgs) < 16:
                imgs += [x[i][0].numpy() for i in range(min(16 - len(imgs), len(x)))]
            correct += (p == y).sum().item(); total += len(y)
    acc = correct / total
    print(f"Test accuracy: {acc:.4f}")
    torch.save(model.state_dict(), ASSETS / "cnn_mnist.pt")
    with open(ASSETS / "metrics.json", "w") as f:
        json.dump({"test_accuracy": acc, "epochs": epochs, "device": DEVICE}, f, indent=2)

    # Sample grid
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(f"true={trues[i]} pred={preds[i]}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"MNIST CNN — acc {acc:.3f}")
    plt.tight_layout()
    plt.savefig(ASSETS / "samples.png", dpi=120)
    plt.close()
    print("Saved cnn_mnist.pt + metrics.json + samples.png")


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(epochs=ep)
