# Face Liveness Detection — CASIA-FASD

A face anti-spoofing classifier trained on the [CASIA Face Anti-Spoofing Dataset (FASD)](http://www.cbsr.ia.ac.cn/english/Databases.asp). It extracts handcrafted features from face images and trains a Multi-Layer Perceptron (MLP) to distinguish real (live) faces from spoofed ones (printed photos, replayed video).

## How it works

1. Each face image is preprocessed with Single-Scale Retinex to normalize lighting
2. Features are extracted using one of five modes:
   - `nn_gradients` — Sobel edge gradient magnitudes
   - `nn_raw_gray` — raw grayscale pixel values
   - `nn_spatial_lbp` — Local Binary Pattern texture descriptors
   - `nn_high_freq` — Laplacian high-frequency response
   - `nn_lpq` — Local Phase Quantization histogram
3. The MLP classifier (`networkb.py`) trains on the extracted features and outputs a live/spoof prediction
4. Performance is measured using APCER, BPCER, and ACER — standard face anti-spoofing metrics

## Project structure

```
.
├── main_nn.py          # Feature extraction + evaluation pipeline
├── networkb.py         # MLP model definition and classifier
├── requirements.txt    # Python dependencies
├── casia-fasd/         # Dataset (train/test splits with live/spoof subfolders)
├── extracted_features/ # Cached .npz feature files (auto-generated)
└── models/             # Saved model checkpoints (auto-generated)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset structure

Download the dataset from [Google Drive](https://drive.google.com/drive/u/1/folders/1bk3JFjH0XaHWPFKoxVm7gY7Cze29CXNz) and place it in `casia-fasd/` with this layout:

```
casia-fasd/
├── train/
│   ├── live/
│   └── spoof/
└── test/
    ├── live/
    └── spoof/
```

## Run

Open `main_nn.py` and set `CHOSEN_MODE` to your desired feature type, then:

```bash
python3 main_nn.py
```

Results print APCER, BPCER, and ACER to the console. Trained models are saved to `models/`.

## Metrics

| Metric | Meaning |
|--------|---------|
| APCER | Attack Presentation Classification Error Rate — spoof classified as live |
| BPCER | Bona-fide Presentation Classification Error Rate — live classified as spoof |
| ACER | Average of APCER and BPCER — overall error |

## Dependencies

- Python 3.9+
- PyTorch
- NumPy
- OpenCV
- scikit-image
- SciPy
- tqdm
