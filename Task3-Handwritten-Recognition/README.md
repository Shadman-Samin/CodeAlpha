# Handwritten Character Recognition — CodeAlpha ML Task 3

Student ID: `CA/DF1/298776`

## Objective
Identify handwritten characters/digits via CNN. Extendable to CRNN for words.

## Dataset
**MNIST** (60k train / 10k test, 28x28 grayscale) via torchvision — auto-downloaded to `data/`.
Compatible with EMNIST-letters (same loader, change `datasets.EMNIST(split='letters')`).

## Model
SmallCNN: Conv(1->32) -> Pool -> Conv(32->64) -> Pool -> FC(3136->128) -> Dropout -> FC(128->10). Adam 1e-3.

## Results (verified on this PC, CUDA)
- 3 epochs: train acc 0.983, **test acc 0.9895**
- Artifacts: `assets/cnn_mnist.pt`, `assets/metrics.json`, `assets/samples.png`

## Run
```powershell
pip install -r requirements.txt
python src/train.py 3
# python src/train.py 10  # higher accuracy (~99.2%)
```

## Submit
GitHub repo: `CodeAlpha_HandwrittenRecognition`
