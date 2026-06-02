import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from lime import lime_tabular
from main_nn import process_cropped_image, CASIAEvaluator
from networkb import MLPLivenessClassifier

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = "./casia-fasd"
MODES = ['nn_gradients', 'nn_raw_gray', 'nn_spatial_lbp', 'nn_high_freq', 'nn_lpq']
NUM_EXPLAIN_SAMPLES = 5
NUM_LIME_FEATURES = 50   # top features LIME selects per explanation
# ─────────────────────────────────────────────────────────────────────────────


def load_sample_images(split, category, n=NUM_EXPLAIN_SAMPLES):
    folder = os.path.join(DATASET_PATH, split, category)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    return files[:n]


def make_predict_fn(model, device):
    """Return [[P(spoof), P(live)]] per sample — required by LIME's classification API."""
    def predict(X):
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(tensor)
            p_live = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.stack([1.0 - p_live, p_live], axis=1)
    return predict


def lime_weights_to_array(explanation, label_idx, feature_dim):
    """Unpack LIME's sparse [(feature_index, weight)] list into a dense vector."""
    arr = np.zeros(feature_dim)
    for feat_idx, weight in explanation.local_exp[label_idx]:
        arr[feat_idx] = weight
    return arr


def explain_mode(mode):
    print(f"\n{'='*60}")
    print(f"  LIME XAI — Mode: {mode}")
    print(f"{'='*60}")

    model_path = f"models/liveness_{mode}.pth"
    if not os.path.exists(model_path):
        print(f"  [!] No saved model at {model_path}. Run main_nn.py first.")
        return

    classifier = MLPLivenessClassifier.load_model(model_path)
    model = classifier.model
    device = classifier.device
    predict_fn = make_predict_fn(model, device)

    # Load training features to anchor the LIME neighbourhood distribution
    evaluator = CASIAEvaluator(DATASET_PATH, mode)
    X_train, _ = evaluator._load_features('train')
    feature_dim = X_train.shape[1]

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"f_{i}" for i in range(feature_dim)],
        class_names=['spoof', 'live'],
        mode='classification',
        discretize_continuous=True,
        random_state=42,
    )

    live_paths  = load_sample_images('test', 'live')
    spoof_paths = load_sample_images('test', 'spoof')
    paths  = live_paths  + spoof_paths
    labels = ['live'] * len(live_paths) + ['spoof'] * len(spoof_paths)

    out_dir = f"xai_output/lime/{mode}"
    os.makedirs(out_dir, exist_ok=True)

    for img_path, true_label in zip(paths, labels):
        feat = process_cropped_image(img_path, mode)
        if feat is None:
            continue

        explanation = explainer.explain_instance(
            feat,
            predict_fn,
            num_features=NUM_LIME_FEATURES,
            num_samples=1000,
            labels=(1,),   # explain class 1 (live)
        )

        pred_prob = predict_fn(feat.reshape(1, -1))[0, 1]
        pred_label = "live" if pred_prob >= 0.5 else "spoof"

        # Dense weight array for class 1 (live); positive = supports live, negative = supports spoof
        lime_arr = lime_weights_to_array(explanation, label_idx=1, feature_dim=feature_dim)

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if mode == 'nn_lpq':
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].imshow(orig_img)
            axes[0].set_title(f"True: {true_label} | Pred: {pred_label} ({pred_prob:.2f})")
            axes[0].axis('off')

            colors = ['green' if v > 0 else 'red' for v in lime_arr]
            axes[1].bar(range(feature_dim), lime_arr, color=colors)
            axes[1].set_title("LIME weights (LPQ bins)\nGreen=supports live, Red=supports spoof")
            axes[1].set_xlabel("LPQ bin index")
            axes[1].set_ylabel("LIME weight")

        else:
            # 4096-dim → 64×64 signed heatmap
            heatmap = lime_arr.reshape(64, 64)
            abs_max = np.abs(heatmap).max() + 1e-8
            heatmap_norm = heatmap / abs_max   # range [-1, 1]

            # PiYG: green = positive (live signal), purple = negative (spoof signal)
            heatmap_color = (plt.cm.PiYG((heatmap_norm + 1) / 2)[:, :, :3] * 255).astype(np.uint8)

            h, w = orig_img.shape[:2]
            hm_resized = cv2.resize(heatmap_color, (w, h))
            overlay = cv2.addWeighted(orig_img, 0.5, hm_resized, 0.5, 0)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\nTrue: {true_label}")
            axes[0].axis('off')

            axes[1].imshow(heatmap_color)
            axes[1].set_title("LIME Heatmap\nGreen=live signal, Purple=spoof signal")
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {pred_label} ({pred_prob:.2f})")
            axes[2].axis('off')

        fig.suptitle(f"LIME XAI — Mode: {mode} | {os.path.basename(img_path)}", fontsize=12)
        plt.tight_layout()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{out_dir}/{true_label}_{stem}.png"
        plt.savefig(out_name, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_name}")


if __name__ == "__main__":
    for mode in MODES:
        explain_mode(mode)
    print("\nDone. Check xai_output/lime/ for explanations.")
