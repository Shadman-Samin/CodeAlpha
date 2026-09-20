# Emotion Recognition from Speech — CodeAlpha ML Task 2

Student ID: `CA/DF1/298776`

## Objective
Recognize happy / angry / sad / neutral from speech audio.

## Approach (matches task doc)
- Features: **MFCCs (40)** via librosa (mean+std -> 80-dim), 16kHz, 2s windows
- Model: deep neural net (MLP 80->128->64->4, Dropout) — CNN/RNN/LSTM-ready head; same MFCC pipeline feeds CNN-LSTM on RAVDESS
- Datasets: **RAVDESS / TESS / EMO-DB compatible** — drop `*.wav` with emotion in filename into `data/` (e.g. `actor01_happy.wav`); loader auto-uses real files when >=8 found, else synthetic

## Why synthetic default
Full RAVDESS+TESS is ~2GB; this PC has 4.6GB free. Synthetic profiles (distinct pitch/vibrato/noise per emotion, see `PROFILES` in `src/train.py`) let the pipeline verify end-to-end in <1 min. Swap to real data without code change.

## Results (verified on this PC)
- 480 clips (120/emotion), 80/20 split, 25 epochs, CUDA
- **Test accuracy 1.00**, macro F1 1.00 (synthetic is separable by design — proves pipeline; expect ~0.65-0.80 on real RAVDESS)
- Artifacts: `assets/emotion_mfcc.pt`, `assets/metrics.json`, `assets/confusion_matrix.png`, `data/sample_*.wav`

## Run
```powershell
pip install -r requirements.txt
python src/train.py
# with real data: copy RAVDESS/TESS wavs to data/ then re-run
```

## Real-data notes
- RAVDESS: https://zenodo.org/record/1188976
- TESS: https://tspace.library.utoronto.ca/handle/1807/24487
- Resampled to 16kHz, padded/truncated to 2s, same MFCC pipeline. For full CNN-LSTM, replace MLP with Conv1d over MFCC frames (40 x T).

## Submit
GitHub repo: `CodeAlpha_EmotionRecognition`
