import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from lime import lime_tabular

from main_nn import process_cropped_image, CASIAEvaluator
from networkb import MLPLivenessClassifier
from xai_gradcam import GradCAMForMLP, _normalize

DATASET_PATH = "./casia-fasd"
MODES = ['nn_gradients', 'nn_raw_gray', 'nn_spatial_lbp', 'nn_high_freq', 'nn_lpq']
SHAP_BACKGROUND = 100
SHAP_NSAMPLES = 200
LIME_NSAMPLES = 1000
LIME_FEATURES = 50
OUT_DIR = "xai_output/comparison"


def _load_one_image(split, category):
    folder = os.path.join(DATASET_PATH, split, category)
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])
    return files[0] if files else None


def _shap_predict_fn(model, device):
    def fn(X):
        t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.sigmoid(model(t)).cpu().numpy().flatten()
    return fn


def _lime_predict_fn(model, device):
    def fn(X):
        t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(t)).cpu().numpy().flatten()
        return np.stack([1 - p, p], axis=1)
    return fn


def _colorize_1d(arr, shape2d, cmap='jet'):
    norm = _normalize(arr.reshape(shape2d))
    return (plt.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)


def _overlay(orig, color_map, shape2d):
    h, w = orig.shape[:2]
    resized = cv2.resize(color_map.reshape(shape2d[0], shape2d[1], 3), (w, h))
    return cv2.addWeighted(orig, 0.5, resized, 0.5, 0)


def compare_mode(mode):
    print(f"\n{'='*60}")
    print(f"  XAI Comparison — Mode: {mode}")
    print(f"{'='*60}")

    model_path = f"models/liveness_{mode}.pth"
    if not os.path.exists(model_path):
        print(f"  [!] No model at {model_path}. Run main_nn.py first.")
        return

    classifier = MLPLivenessClassifier.load_model(model_path)
    model = classifier.model
    device = classifier.device

    evaluator = CASIAEvaluator(DATASET_PATH, mode)
    X_train, _ = evaluator._load_features('train')
    feature_dim = X_train.shape[1]
    is_spatial = (feature_dim == 4096)

    shap_predict = _shap_predict_fn(model, device)
    lime_predict = _lime_predict_fn(model, device)

    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), size=SHAP_BACKGROUND, replace=False)
    shap_explainer = shap.KernelExplainer(shap_predict, X_train[bg_idx])

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f"f_{i}" for i in range(feature_dim)],
        class_names=['spoof', 'live'],
        mode='classification',
        discretize_continuous=True,
        random_state=42,
    )

    gradcam = GradCAMForMLP(model)
    out_dir = os.path.join(OUT_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)

    for category in ('live', 'spoof'):
        img_path = _load_one_image('test', category)
        if img_path is None:
            continue
        feat = process_cropped_image(img_path, mode)
        if feat is None:
            continue

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Grad-CAM
        x_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        cam_arr, _, pred_prob = gradcam.compute(x_t)
        pred_label = "live" if pred_prob >= 0.5 else "spoof"

        # SHAP
        shap_vals = shap_explainer.shap_values(feat.reshape(1, -1), nsamples=SHAP_NSAMPLES)
        shap_arr = shap_vals[0]

        # LIME
        exp = lime_explainer.explain_instance(
            feat, lime_predict,
            num_features=LIME_FEATURES, num_samples=LIME_NSAMPLES, labels=(1,)
        )
        lime_arr = np.zeros(feature_dim)
        for idx, w in exp.local_exp[1]:
            lime_arr[idx] = w

        if is_spatial:
            shape2d = (64, 64)
            cam_color  = _colorize_1d(np.abs(cam_arr), shape2d, cmap='jet')
            shap_color = _colorize_1d(np.abs(shap_arr), shape2d, cmap='RdBu_r')
            lime_color = _colorize_1d(np.abs(lime_arr), shape2d, cmap='PiYG')

            h, w = orig_img.shape[:2]
            def blend(c): return cv2.addWeighted(orig_img, 0.5, cv2.resize(c, (w, h)), 0.5, 0)

            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            for ax in axes:
                ax.axis('off')
            axes[0].imshow(orig_img)
            axes[0].set_title(f"Original\nTrue: {category} | Pred: {pred_label} ({pred_prob:.2f})")
            axes[1].imshow(blend(cam_color))
            axes[1].set_title("Grad-CAM\n(gradient × activation, projected to input)")
            axes[2].imshow(blend(shap_color))
            axes[2].set_title("SHAP\n(Shapley attribution vs background)")
            axes[3].imshow(blend(lime_color))
            axes[3].set_title("LIME\n(local linear surrogate weights)")

        else:
            fig, axes = plt.subplots(1, 4, figsize=(22, 4))
            axes[0].imshow(orig_img); axes[0].axis('off')
            axes[0].set_title(f"Original\nTrue: {category} | Pred: {pred_label} ({pred_prob:.2f})")

            axes[1].bar(range(len(cam_arr)), np.abs(cam_arr), color='steelblue')
            axes[1].set_title("Grad-CAM importance"); axes[1].set_xlabel("LPQ bin")

            axes[2].bar(range(len(shap_arr)), shap_arr,
                        color=['red' if v > 0 else 'blue' for v in shap_arr])
            axes[2].set_title("SHAP values\nRed=spoof signal, Blue=live signal")
            axes[2].set_xlabel("LPQ bin")

            axes[3].bar(range(len(lime_arr)), lime_arr,
                        color=['green' if v > 0 else 'purple' for v in lime_arr])
            axes[3].set_title("LIME weights\nGreen=live signal, Purple=spoof signal")
            axes[3].set_xlabel("LPQ bin")

        fig.suptitle(f"XAI Comparison — Mode: {mode} | {category}", fontsize=12)
        plt.tight_layout()
        out_path = f"{out_dir}/{category}_comparison.png"
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    gradcam.remove_hooks()


ANALYSIS_REPORT = """
╔══════════════════════════════════════════════════════════════════╗
║         XAI METHOD COMPARATIVE ANALYSIS                         ║
║         Liveness Detection — CASIA-FASD Dataset                 ║
╚══════════════════════════════════════════════════════════════════╝

Three explainability methods were applied to the trained LivenessNet
MLP classifier across all five feature extraction modes (nn_gradients,
nn_raw_gray, nn_spatial_lbp, nn_high_freq, nn_lpq).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GRAD-CAM  (Gradient-weighted Class Activation Mapping)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What it extracts:
  Grad-CAM computes the gradient of the output with respect to the
  last hidden layer (32 units), alpha-weights activations by those
  gradients, and back-projects the result through the MLP weight
  matrices (W6→W3→W0) to the input space.  This yields a per-input-
  feature importance map showing which regions of the feature vector
  drove the live/spoof decision at the final hidden layer.

Advantages:
  + Fast — single forward + backward pass, no extra model queries.
  + Faithful — uses actual model gradients, not a surrogate.
  + Deterministic — same input always produces the same map.
  + Captures which features the penultimate representation relied on.

Drawbacks:
  - MLP adaptation requires manual weight-chain back-projection;
    true CNNs have native spatial feature maps that map more cleanly.
  - Back-projected maps can be diffuse because MLP linear layers
    mix all input regions during forward computation.
  - Only reflects the last hidden layer; early-layer specialisation
    is not directly visible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. SHAP  (SHapley Additive exPlanations — KernelSHAP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What it extracts:
  SHAP assigns each feature a Shapley value — its average marginal
  contribution to the output across all possible feature coalitions,
  measured against a background distribution of 100 training samples.
  Positive values push the prediction toward spoof; negative values
  toward live (since P(live) is the explained quantity).

Advantages:
  + Theoretically grounded: satisfies efficiency, symmetry, dummy,
    and additivity axioms from cooperative game theory.
  + Model-agnostic — works with any black-box callable.
  + Signed values distinguish features that help vs. hurt liveness.
  + Can aggregate over many samples for global feature importance.

Drawbacks:
  - Slowest method — KernelSHAP runs 200 coalition evaluations per
    sample, making it impractical at scale without approximation.
  - Results depend on the background distribution; a biased baseline
    leads to biased attributions.
  - Assumes feature independence within coalitions, which may not
    hold for spatially correlated pixel-level features.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. LIME  (Local Interpretable Model-agnostic Explanations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What it extracts:
  LIME perturbs the input sample 1,000 times in its neighbourhood
  (Gaussian noise around discretised feature bins), queries the MLP
  on each perturbed sample, then fits a sparse weighted linear model
  to the query results.  The top-50 linear coefficients reveal which
  features a locally linear approximation of the MLP relies on near
  that specific sample.

Advantages:
  + Model-agnostic — treats the MLP as a pure black box.
  + Intuitive output: linear weights are easy to communicate.
  + Moderate speed — faster than SHAP for high-dimensional data when
    using a small number of top features.
  + Flexible: the same framework applies to tabular, image, and text.

Drawbacks:
  - Locally faithful only — the linear surrogate may break down
    outside the immediate neighbourhood of the sample.
  - Non-deterministic — different random seeds yield different
    neighbourhood samples and therefore different weight values.
  - Neighbourhood definition (Gaussian perturbation of discretised
    bins) is an approximation that may not match the true data manifold.
  - Can miss complex non-linear feature interactions that Grad-CAM
    and SHAP capture more directly through model internals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY TABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Property                  Grad-CAM    SHAP        LIME
  ────────────────────────  ────────    ────        ────
  Speed                     Fast        Slow        Medium
  Model-agnostic            No          Yes         Yes
  Theoretically grounded    Partial     Yes         Partial
  Deterministic             Yes         Yes         No
  Signed attribution        No          Yes         Yes
  Captures non-linearity    Yes         Yes         No
  Global feature insight    No          Yes         No
  Computational cost        1 pass      O(coalitions) O(samples)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For this liveness detection study:

  • SHAP is the most principled and information-rich method.  Its
    Shapley values are theoretically guaranteed to be the unique fair
    attribution satisfying all four axioms, making it the gold
    standard for reporting in academic work.  Use it for the final
    analysis on selected representative samples.

  • Grad-CAM is the most practical for rapid, large-scale inspection.
    It requires only one gradient computation and is fully
    deterministic.  For the four spatial feature modes (gradients,
    raw gray, LBP, high-freq), the 64×64 back-projected heatmap gives
    a clear visual intuition of which image regions the network
    found discriminative.

  • LIME serves as a cross-validation tool.  When LIME and SHAP agree
    on the top contributing features, confidence in the explanation
    is higher.  When they disagree, Grad-CAM's gradient-based view
    (grounded in actual model internals) should be treated as the
    more reliable signal.

  Recommended workflow:
    Train model → Grad-CAM (rapid screening of all test samples)
    → SHAP (deep per-sample analysis for report figures)
    → LIME (sanity-check agreement with SHAP on key samples)
"""


def save_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    report_path = os.path.join(OUT_DIR, "xai_comparative_analysis.txt")
    with open(report_path, "w") as f:
        f.write(ANALYSIS_REPORT)
    print(ANALYSIS_REPORT)
    print(f"  Report saved → {report_path}")


if __name__ == "__main__":
    for mode in MODES:
        compare_mode(mode)
    save_report()
    print("\nDone. Check xai_output/comparison/ for figures and report.")
