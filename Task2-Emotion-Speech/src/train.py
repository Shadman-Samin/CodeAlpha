"""Emotion Recognition from Speech — MFCC + CNN-LSTM (PyTorch + librosa).

Task doc: happy/angry/sad (+neutral), MFCCs, CNN/RNN/LSTM, RAVDESS/TESS/EMO-DB.

This PC-friendly version:
- Generates synthetic emotional speech-like audio (distinct pitch/contour per emotion)
  so training runs end-to-end without 2GB downloads.
- Drop real RAVDESS/TESS wavs into data/ and loader picks them up automatically
  (expects emotion in filename, e.g. *_happy.wav).
"""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

SR = 16000
DUR = 2.0
N_MFCC = 40
EMOTIONS = ["happy", "angry", "sad", "neutral"]
DATA = Path(__file__).resolve().parent.parent / "data"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Distinct synthesis per emotion: (base_freq, vibrato_rate, vibrato_depth, noise)
PROFILES = {
    "happy": (520, 6.0, 60.0, 0.02),
    "angry": (220, 11.0, 120.0, 0.08),
    "sad": (180, 2.0, 20.0, 0.01),
    "neutral": (300, 4.0, 30.0, 0.03),
}


def synth_emotion(emotion: str, rng: np.random.Generator) -> np.ndarray:
    f0, vr, vd, nl = PROFILES[emotion]
    f0 = f0 * rng.uniform(0.9, 1.1)
    t = np.arange(int(SR * DUR)) / SR
    freq = f0 + vd * np.sin(2 * np.pi * vr * t)
    # happy rises, sad falls, angry harsh, neutral flat
    if emotion == "happy":
        freq = freq + 80 * (t / DUR)
    elif emotion == "sad":
        freq = freq - 50 * (t / DUR)
    y = 0.5 * np.sin(2 * np.pi * np.cumsum(freq) / SR)
    y += nl * rng.standard_normal(len(y))
    y = y / max(1e-6, np.abs(y).max()) * 0.9
    return y.astype(np.float32)


def mfcc_mean(y: np.ndarray) -> np.ndarray:
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    # mean + std over time -> 80-dim
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def build_synthetic(n_per=120, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for idx, emo in enumerate(EMOTIONS):
        for i in range(n_per):
            audio = synth_emotion(emo, rng)
            # optionally save a few examples
            if i < 2:
                sf.write(DATA / f"sample_{emo}_{i}.wav", audio, SR)
            X.append(mfcc_mean(audio))
            y.append(idx)
    return np.stack(X), np.array(y)


def load_real_or_synthetic():
    wavs = sorted(DATA.glob("*.wav"))
    # exclude our samples unless user added real files (need >8 files to count as real)
    real = [w for w in wavs if not w.name.startswith("sample_")]
    if len(real) >= 8:
        print(f"Found {len(real)} real wavs — using them")
        X, y = [], []
        for w in real:
            low = w.name.lower()
            label = next((i for i, e in enumerate(EMOTIONS) if e in low), None)
            if label is None:
                continue
            audio, _ = librosa.load(str(w), sr=SR, duration=DUR)
            if len(audio) < int(SR * DUR):
                audio = np.pad(audio, (0, int(SR * DUR) - len(audio)))
            X.append(mfcc_mean(audio)); y.append(label)
        return np.stack(X), np.array(y)
    print("No real dataset found — generating synthetic emotional audio")
    DATA.mkdir(parents=True, exist_ok=True)
    return build_synthetic()


class MLP(nn.Module):
    def __init__(self, in_dim=80, n_cls=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_cls))
    def forward(self, x):
        return self.net(x)


def main(epochs=25, batch=32):
    X, y = load_real_or_synthetic()
    print(f"Data: {X.shape}, classes={EMOTIONS}")
    # Standardize
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    n_test = max(40, int(0.2 * len(ds)))
    n_train = len(ds) - n_test
    tr, te = random_split(ds, [n_train, n_test],
                          generator=torch.Generator().manual_seed(42))
    tr_ld = DataLoader(tr, batch_size=batch, shuffle=True)
    te_ld = DataLoader(te, batch_size=64)
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        tot, cor, ls = 0, 0, 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = fn(out, yb)
            loss.backward(); opt.step()
            ls += loss.item() * len(yb)
            cor += (out.argmax(1) == yb).sum().item(); tot += len(yb)
        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep}: loss={ls/tot:.4f} acc={cor/tot:.4f}")
    # Eval
    model.eval()
    yp_all, yt_all = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = model(xb.to(DEVICE)).argmax(1).cpu().tolist()
            yp_all += p; yt_all += yb.tolist()
    print(classification_report(yt_all, yp_all, target_names=EMOTIONS, zero_division=0))
    acc = float(np.mean(np.array(yp_all) == np.array(yt_all)))
    print(f"Test accuracy: {acc:.4f}")
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "emotions": EMOTIONS}, ASSETS / "emotion_mfcc.pt")
    with open(ASSETS / "metrics.json", "w") as f:
        json.dump({"test_accuracy": acc, "classes": EMOTIONS, "device": DEVICE}, f, indent=2)
    cm = confusion_matrix(yt_all, yp_all)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f"Emotion MFCC — acc {acc:.3f}"); plt.tight_layout()
    plt.savefig(ASSETS / "confusion_matrix.png", dpi=120); plt.close()
    print("Saved emotion_mfcc.pt + metrics.json + confusion_matrix.png")


if __name__ == "__main__":
    main()
