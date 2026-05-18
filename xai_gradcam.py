import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from main_nn import process_cropped_image
from networkb import MLPLivenessClassifier, LivenessNet

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH = "./casia-fasd"
MODES = ['nn_gradients', 'nn_raw_gray', 'nn_spatial_lbp', 'nn_high_freq', 'nn_lpq']
NUM_EXPLAIN_SAMPLES = 5
# ─────────────────────────────────────────────────────────────────────────────


def load_sample_images(split, category, n=NUM_EXPLAIN_SAMPLES):
    folder = os.path.join(DATASET_PATH, split, category)
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    return files[:n]


class GradCAMForMLP:
    """
    Grad-CAM adapted for LivenessNet (MLP).

    CNN Grad-CAM:  hooks last conv layer → alpha-weighted feature maps → upsample.
    MLP Grad-CAM:  hooks last hidden layer (network[7], 32 units) → alpha-weighted
                   activations → project back to input space via transposed weight
                   matrices, yielding a (input_dim,) importance vector.
    For the four 64×64 spatial modes this reshapes directly to a 64×64 heatmap.

    LivenessNet.network indices:
        [0] Linear(input→128)  [1] LeakyReLU  [2] Dropout
        [3] Linear(128→64)     [4] LeakyReLU  [5] Dropout
        [6] Linear(64→32)      [7] LeakyReLU  [8] Dropout
        [9] Linear(32→1)
    """

    def __init__(self, model: LivenessNet):
        self.model = model
        self.model.eval()
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.network[7]   # LeakyReLU after 3rd Linear (output dim=32)

        def fwd_hook(module, inp, out):
            self._acts = out

        def bwd_hook(module, grad_in, grad_out):
            self._grads = grad_out[0]

        self._handles.append(target.register_forward_hook(fwd_hook))
        self._handles.append(target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def compute(self, x_tensor: torch.Tensor):
        """
        Forward + backward pass.  Returns:
          cam_input  (np.ndarray, input_dim): GradCAM importance projected to input
          saliency   (np.ndarray, input_dim): |∂output/∂input| vanilla saliency
          pred_prob  (float): sigmoid probability of 'live'
        """
        self.model.zero_grad()
        x = x_tensor.clone().float().requires_grad_(True)

        logit = self.model(x)
        pred_prob = torch.sigmoid(logit).item()
        logit.backward()

        # ── Vanilla gradient saliency: |∂output/∂input| ───────────────────
        saliency = x.grad.abs().squeeze(0).detach().cpu().numpy()

        # ── GradCAM at last hidden layer ───────────────────────────────────
        # alpha_k = gradient of output w.r.t. k-th hidden unit (GradCAM weight)
        # For 1-D feature vectors there is no spatial GAP step; alpha = gradient directly.
        alpha = self._grads.squeeze(0).detach().cpu()   # (32,)
        acts  = self._acts.squeeze(0).detach().cpu()    # (32,)
        cam_hidden = torch.relu(alpha * acts)            # (32,) — Grad-CAM formula

        # ── Project cam_hidden back to input space via W^T chain ───────────
        # nn.Linear weight shape: (out_features, in_features)
        # so weight.T has shape:  (in_features, out_features)
        # network[6]: Linear(64→32) → W6.T: (64,32)  applied to (32,) → (64,)
        # network[3]: Linear(128→64)→ W3.T: (128,64) applied to (64,) → (128,)
        # network[0]: Linear(in→128)→ W0.T: (in,128) applied to (128,) → (in,)
        W6_T = self.model.network[6].weight.detach().cpu().T   # (64, 32)
        W3_T = self.model.network[3].weight.detach().cpu().T   # (128, 64)
        W0_T = self.model.network[0].weight.detach().cpu().T   # (input_dim, 128)

        proj = W6_T @ cam_hidden   # (64,)
        proj = W3_T @ proj          # (128,)
        cam_input = (W0_T @ proj).numpy()   # (input_dim,)
        cam_input = np.abs(cam_input)

        return cam_input, saliency, pred_prob


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _colorize(heatmap_2d: np.ndarray, orig_img: np.ndarray):
    """Return (jet_heatmap_uint8_rgb, alpha_blend_overlay) from a [0,1] 2D heatmap."""
    h, w = orig_img.shape[:2]
    hm = cv2.resize(heatmap_2d.astype(np.float32), (w, h))
    hm_color = (plt.cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(orig_img, 0.5, hm_color, 0.5, 0)
    return hm_color, overlay


# ── Per-mode explanation ─────────────────────────────────────────────────────

def explain_mode_gradcam(mode: str):
    print(f"\n{'='*60}")
    print(f"  Grad-CAM XAI — Mode: {mode}")
    print(f"{'='*60}")

    model_path = f"models/liveness_{mode}.pth"
    if not os.path.exists(model_path):
        print(f"  [!] No saved model at {model_path}. Run main_nn.py first.")
        return

    classifier = MLPLivenessClassifier.load_model(model_path)
    gradcam = GradCAMForMLP(classifier.model)
    device = classifier.device

    live_paths  = load_sample_images('test', 'live')
    spoof_paths = load_sample_images('test', 'spoof')
    paths  = live_paths  + spoof_paths
    labels = ['live'] * len(live_paths) + ['spoof'] * len(spoof_paths)

    out_dir = f"xai_output/gradcam/{mode}"
    os.makedirs(out_dir, exist_ok=True)

    for img_path, true_label in zip(paths, labels):
        feat = process_cropped_image(img_path, mode)
        if feat is None:
            continue

        x_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        cam_input, saliency, pred_prob = gradcam.compute(x_tensor)
        pred_label = "live" if pred_prob >= 0.5 else "spoof"
        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if mode == 'nn_lpq':
            # LPQ outputs a 256-dim histogram — no spatial reshape possible, use bar charts
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            axes[0].imshow(orig_img)
            axes[0].axis('off')
            axes[0].set_title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})")

            c_colors = plt.cm.jet(_normalize(cam_input))
            axes[1].bar(range(len(cam_input)), cam_input, color=c_colors)
            axes[1].set_title("Grad-CAM importance (LPQ bins)")
            axes[1].set_xlabel("Bin index")
            axes[1].set_ylabel("Importance")

            s_colors = plt.cm.plasma(_normalize(saliency))
            axes[2].bar(range(len(saliency)), saliency, color=s_colors)
            axes[2].set_title("Gradient Saliency (LPQ bins)")
            axes[2].set_xlabel("Bin index")

        else:
            # Spatial modes: 4096 features → 64×64 heatmap
            cam_2d = _normalize(cam_input.reshape(64, 64))
            sal_2d = _normalize(saliency.reshape(64, 64))

            cam_color, cam_overlay = _colorize(cam_2d, orig_img)
            sal_color, sal_overlay = _colorize(sal_2d, orig_img)

            fig, axes = plt.subplots(2, 3, figsize=(16, 9))

            # Row 0: Grad-CAM (last hidden layer → input projection)
            axes[0, 0].imshow(orig_img)
            axes[0, 0].axis('off')
            axes[0, 0].set_title(f"Original\nTrue: {true_label}")

            axes[0, 1].imshow(cam_color)
            axes[0, 1].axis('off')
            axes[0, 1].set_title("Grad-CAM heatmap\n(last hidden layer projected to input)")

            axes[0, 2].imshow(cam_overlay)
            axes[0, 2].axis('off')
            axes[0, 2].set_title(f"Grad-CAM overlay\nPred: {pred_label} ({pred_prob:.2f})")

            # Row 1: Vanilla gradient saliency for comparison
            axes[1, 0].imshow(orig_img)
            axes[1, 0].axis('off')
            axes[1, 0].set_title("Original")

            axes[1, 1].imshow(sal_color)
            axes[1, 1].axis('off')
            axes[1, 1].set_title("Gradient Saliency heatmap\n|∂output / ∂input|")

            axes[1, 2].imshow(sal_overlay)
            axes[1, 2].axis('off')
            axes[1, 2].set_title("Gradient Saliency overlay")

        fig.suptitle(
            f"Grad-CAM vs Gradient Saliency — Mode: {mode} | {os.path.basename(img_path)}",
            fontsize=11
        )
        plt.tight_layout()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{out_dir}/{true_label}_{stem}.png"
        plt.savefig(out_name, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_name}")

    gradcam.remove_hooks()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for mode in MODES:
        explain_mode_gradcam(mode)
    print("\nDone. Check xai_output/gradcam/ for heatmaps.")
