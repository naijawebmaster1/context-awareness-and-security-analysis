import os
import cv2
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
from main_nn import process_cropped_image, CASIAEvaluator
from networkb import MLPLivenessClassifier, LivenessNet

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH = "./casia-fasd"
MODES = ['nn_gradients', 'nn_raw_gray', 'nn_spatial_lbp', 'nn_high_freq', 'nn_lpq']
NUM_EXPLAIN_SAMPLES = 5   # how many live + spoof images to explain per mode
# ────────────────────────────────────────────────────────────────────────────

def load_sample_images(split, category, n=NUM_EXPLAIN_SAMPLES):
    """Return n image paths from dataset_path/split/category."""
    folder = os.path.join(DATASET_PATH, split, category)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    return files[:n]

def make_predict_fn(model, device):
    """Wrap the PyTorch model so SHAP can call it like a function."""
    def predict(X):
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        # return probability of LIVE (class 1) for each sample
        return probs.flatten()
    return predict

def explain_mode(mode):
    print(f"\n{'='*60}")
    print(f"  XAI Analysis — Mode: {mode}")
    print(f"{'='*60}")

    # Load trained model for this mode
    model_path = f"models/liveness_{mode}.pth"
    if not os.path.exists(model_path):
        print(f"  [!] No saved model found at {model_path}. Run main_nn.py first.")
        return

    classifier = MLPLivenessClassifier.load_model(model_path)
    model = classifier.model
    device = classifier.device
    predict_fn = make_predict_fn(model, device)

    # Load background samples for SHAP (use training live+spoof mix as baseline)
    # SHAP needs a "background" distribution to compare against
    evaluator = CASIAEvaluator(DATASET_PATH, mode)
    X_train, y_train = evaluator._load_features('train')
    background = X_train[np.random.choice(len(X_train), size=100, replace=False)]

    # Build SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, background)

    # Collect test samples to explain
    live_paths = load_sample_images('test', 'live')
    spoof_paths = load_sample_images('test', 'spoof')
    sample_paths = live_paths + spoof_paths
    sample_labels = ['live'] * len(live_paths) + ['spoof'] * len(spoof_paths)

    os.makedirs(f"xai_output/shap/{mode}", exist_ok=True)

    for img_path, true_label in zip(sample_paths, sample_labels):
        feat = process_cropped_image(img_path, mode)
        if feat is None:
            continue

        # Compute SHAP values for this one sample
        shap_values = explainer.shap_values(feat.reshape(1, -1), nsamples=200)
        shap_arr = shap_values[0]   # shape: (feature_dim,)

        # Get model prediction
        pred_prob = predict_fn(feat.reshape(1, -1))[0]
        pred_label = "live" if pred_prob >= 0.5 else "spoof"

        # ── Visualize ──────────────────────────────────────────────────────
        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # For LPQ (histogram, 256-dim) we can't reshape to 64x64, so skip heatmap
        if mode == 'nn_lpq':
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(orig_img)
            axes[0].set_title(f"True: {true_label} | Pred: {pred_label} ({pred_prob:.2f})")
            axes[0].axis('off')

            axes[1].bar(range(len(shap_arr)), shap_arr,
                        color=['red' if v > 0 else 'blue' for v in shap_arr])
            axes[1].set_title("SHAP values (LPQ bins)")
            axes[1].set_xlabel("LPQ bin index")
            axes[1].set_ylabel("SHAP value")

        else:
            # All other modes: reshape 4096 features → 64×64 heatmap
            heatmap = shap_arr.reshape(64, 64)
            heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

            # Normalize to [-1, 1] for display
            abs_max = np.abs(heatmap_resized).max() + 1e-8
            heatmap_norm = heatmap_resized / abs_max  # range [-1, 1]

            # Convert to color: positive=red (spoof signal), negative=blue (live signal)
            heatmap_color = plt.cm.RdBu_r((heatmap_norm + 1) / 2)[:, :, :3]
            heatmap_color = (heatmap_color * 255).astype(np.uint8)

            # Blend overlay onto original
            overlay = cv2.addWeighted(orig_img, 0.5, heatmap_color, 0.5, 0)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\nTrue: {true_label}")
            axes[0].axis('off')

            axes[1].imshow(heatmap_color)
            axes[1].set_title("SHAP Heatmap\nRed=spoof signal, Blue=live signal")
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {pred_label} ({pred_prob:.2f})")
            axes[2].axis('off')

        fig.suptitle(f"Mode: {mode} — {os.path.basename(img_path)}", fontsize=12)
        plt.tight_layout()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"xai_output/shap/{mode}/{true_label}_{stem}.png"
        plt.savefig(out_name, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_name}")

if __name__ == "__main__":
    for mode in MODES:
        explain_mode(mode)

    print("\nDone. Check the xai_output/ folder for heatmaps.")
