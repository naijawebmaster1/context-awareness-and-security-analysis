import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from main_nn import process_cropped_image, CASIAEvaluator

DATASET_PATH   = "./casia-fasd"
MODES          = ['nn_gradients', 'nn_raw_gray', 'nn_spatial_lbp', 'nn_high_freq', 'nn_lpq']
NUM_POS_SLOTS  = 2
NUM_NEG_SLOTS  = 2
SLOT_DIM       = 32
NUM_ITERS      = 3
EPOCHS         = 30
BATCH_SIZE     = 64
LR             = 5e-4
NUM_SAMPLES    = 5


class SlotAttention(nn.Module):
    """Iterative slot attention (Locatello et al., NeurIPS 2020)."""

    def __init__(self, num_slots: int, slot_dim: int, num_iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim  = slot_dim
        self.num_iters = num_iters
        self.eps       = eps
        self.scale     = slot_dim ** -0.5

        # learnable Gaussian init for slots
        self.slots_mu        = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim * 2, slot_dim),
        )

    def forward(self, inputs: torch.Tensor):
        B, N, _ = inputs.shape
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu    = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn_out = None
        for _ in range(self.num_iters):
            prev = slots
            q    = self.to_q(self.norm_slots(slots))

            # softmax across slots so each position is claimed by its best slot
            dots    = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn    = dots.softmax(dim=1) + self.eps
            attn_out = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bkn,bnd->bkd', attn_out, v)

            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                prev.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            slots = slots + self.mlp(slots)

        return slots, attn_out


class SCOUTERLivenessNet(nn.Module):
    """SCOUTER-style liveness classifier for flat feature vectors."""

    def __init__(
        self,
        feature_dim:   int = 4096,
        num_pos_slots: int = NUM_POS_SLOTS,
        num_neg_slots: int = NUM_NEG_SLOTS,
        slot_dim:      int = SLOT_DIM,
        num_iters:     int = NUM_ITERS,
    ):
        super().__init__()
        self.feature_dim   = feature_dim
        self.num_pos_slots = num_pos_slots
        self.num_neg_slots = num_neg_slots
        total_slots        = num_pos_slots + num_neg_slots

        self.input_proj = nn.Sequential(
            nn.Linear(1, slot_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(slot_dim),
        )

        self.slot_attn  = SlotAttention(total_slots, slot_dim, num_iters)
        self.slot_score = nn.Linear(slot_dim, 1)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        spatial  = x.view(B, self.feature_dim, 1)
        features = self.input_proj(spatial)

        slots, attn = self.slot_attn(features)
        scores = self.slot_score(slots).squeeze(-1)

        pos_score = scores[:, :self.num_pos_slots].mean(dim=1)
        neg_score = scores[:, self.num_pos_slots:].mean(dim=1)
        logit = (pos_score - neg_score).unsqueeze(1)

        return logit, attn


class SCOUTERClassifier:

    def __init__(self, feature_dim: int = 4096):
        self.feature_dim = feature_dim
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = SCOUTERLivenessNet(feature_dim=feature_dim).to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # weighted loss for live/spoof imbalance
        num_spoof  = (y == 0).sum()
        num_live   = (y == 1).sum()
        pos_weight = torch.tensor([num_spoof / (num_live + 1e-7)]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-4)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logit, _ = self.model(bx)
                loss = criterion(logit, by)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [SCOUTER] Epoch {epoch+1:02d}/{EPOCHS} | loss {total/len(loader):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logit, _ = self.model(tensor)
            preds = (torch.sigmoid(logit) >= 0.5).int().cpu().numpy()
        return preds.flatten()

    def explain(self, feat: np.ndarray):
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logit, attn = self.model(x)
        pred_prob = torch.sigmoid(logit).item()
        attn_np   = attn.squeeze(0).cpu().numpy()
        P         = self.model.num_pos_slots
        pos_attn  = attn_np[:P].mean(axis=0)
        neg_attn  = attn_np[P:].mean(axis=0)
        return pred_prob, pos_attn, neg_attn

    def save(self, path: str):
        torch.save({'state': self.model.state_dict(), 'feature_dim': self.feature_dim}, path)

    @classmethod
    def load(cls, path: str) -> 'SCOUTERClassifier':
        ckpt  = torch.load(path, map_location='cpu', weights_only=True)
        inst  = cls(feature_dim=ckpt['feature_dim'])
        inst.model = SCOUTERLivenessNet(feature_dim=ckpt['feature_dim']).to(inst.device)
        inst.model.load_state_dict(ckpt['state'])
        inst.model.eval()
        return inst


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def _load_sample_images(split: str, category: str, n: int = NUM_SAMPLES):
    folder = os.path.join(DATASET_PATH, split, category)
    files  = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])
    return files[:n]


def train_scouter(mode: str) -> SCOUTERClassifier:
    print(f"\n{'='*60}")
    print(f"  SCOUTER — Mode: {mode}")
    print(f"{'='*60}")

    model_path = f"models/scouter_{mode}.pth"
    evaluator  = CASIAEvaluator(DATASET_PATH, mode)
    X_train, y_train = evaluator._load_features('train')
    feature_dim = X_train.shape[1]

    if os.path.exists(model_path):
        print(f"  Loading saved SCOUTER model from {model_path}")
        clf = SCOUTERClassifier.load(model_path)
    else:
        clf = SCOUTERClassifier(feature_dim=feature_dim)
        clf.train(X_train, y_train)
        clf.save(model_path)
        print(f"  Model saved → {model_path}")

    X_test, y_test = evaluator._load_features('test')
    preds = clf.predict(X_test)
    apcer = np.sum((preds == 1) & (y_test == 0)) / np.sum(y_test == 0)
    bpcer = np.sum((preds == 0) & (y_test == 1)) / np.sum(y_test == 1)
    acer  = (apcer + bpcer) * 50
    print(f"  APCER: {apcer*100:.2f}%  BPCER: {bpcer*100:.2f}%  ACER: {acer:.2f}%")

    return clf


def visualize_scouter(mode: str, clf: SCOUTERClassifier):
    print(f"\n  Visualising SCOUTER explanations — Mode: {mode}")
    is_spatial = (clf.feature_dim == 4096)
    out_dir    = f"xai_output/scouter/{mode}"
    os.makedirs(out_dir, exist_ok=True)

    for category in ('live', 'spoof'):
        for img_path in _load_sample_images('test', category):
            feat = process_cropped_image(img_path, mode)
            if feat is None:
                continue

            pred_prob, pos_attn, neg_attn = clf.explain(feat)
            pred_label = "live" if pred_prob >= 0.5 else "spoof"
            orig_img   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            if is_spatial:
                h, w   = orig_img.shape[:2]
                pos_2d = _normalize(pos_attn.reshape(64, 64))
                neg_2d = _normalize(neg_attn.reshape(64, 64))

                def _blend(map_2d, cmap):
                    color = (plt.get_cmap(cmap)(map_2d)[:, :, :3] * 255).astype(np.uint8)
                    return cv2.addWeighted(orig_img, 0.5, cv2.resize(color, (w, h)), 0.5, 0)

                diff_norm = _normalize(pos_2d - neg_2d)

                fig, axes = plt.subplots(1, 4, figsize=(22, 5))
                for ax in axes:
                    ax.axis('off')

                axes[0].imshow(orig_img)
                axes[0].set_title(
                    f"Original\nTrue: {category}  Pred: {pred_label} ({pred_prob:.2f})"
                )
                axes[1].imshow(_blend(pos_2d, 'YlGn'))
                axes[1].set_title("Positive Slots\n(live evidence — brighter = stronger)")
                axes[2].imshow(_blend(neg_2d, 'OrRd'))
                axes[2].set_title("Negative Slots\n(spoof evidence — brighter = stronger)")
                axes[3].imshow(_blend(diff_norm, 'RdYlGn'))
                axes[3].set_title("Net Evidence\nGreen = live  Red = spoof")

            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))

                axes[0].imshow(orig_img)
                axes[0].axis('off')
                axes[0].set_title(
                    f"Original\nTrue: {category}  Pred: {pred_label} ({pred_prob:.2f})"
                )

                pos_colors = plt.cm.YlGn(_normalize(pos_attn))
                axes[1].bar(range(len(pos_attn)), pos_attn, color=pos_colors)
                axes[1].set_title("Positive Slots (live support)")
                axes[1].set_xlabel("LPQ bin index")
                axes[1].set_ylabel("Attention weight")

                neg_colors = plt.cm.OrRd(_normalize(neg_attn))
                axes[2].bar(range(len(neg_attn)), neg_attn, color=neg_colors)
                axes[2].set_title("Negative Slots (spoof support)")
                axes[2].set_xlabel("LPQ bin index")

            fig.suptitle(
                f"SCOUTER Explanations — Mode: {mode} | {os.path.basename(img_path)}",
                fontsize=11,
            )
            plt.tight_layout()

            stem     = os.path.splitext(os.path.basename(img_path))[0]
            out_path = f"{out_dir}/{category}_{stem}.png"
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    for mode in MODES:
        clf = train_scouter(mode)
        visualize_scouter(mode, clf)
    print("\nDone. Check xai_output/scouter/ for explanations.")
