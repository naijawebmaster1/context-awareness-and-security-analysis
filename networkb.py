import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class LivenessNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LivenessNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

class MLPLivenessClassifier:
    def __init__(self, epochs=50, batch_size=32, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X, y, plot_path=None):
        print(f"\n MLP on {self.device}...")

        torch.set_num_threads(os.cpu_count())

        input_dim = X.shape[1]
        self.model = LivenessNet(input_size=input_dim).to(self.device)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        num_spoof = (y == 0).sum()
        num_live = (y == 1).sum()
        weight_ratio = num_spoof / (num_live + 1e-7)
        pos_weight_tensor = torch.tensor([weight_ratio], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[MLP] Training for {self.epochs} epochs")
        self.model.train()
        epoch_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f" Epoch {epoch+1:03d}/{self.epochs} | Average loss: {avg_loss:.4f}")

        if plot_path is not None:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, self.epochs + 1), epoch_losses, linewidth=1.5, color='steelblue')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BCE Loss")
            ax.set_title("Training Loss Curve")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"[MLP] Training curve saved → {plot_path}")

    def predict(self, X_test):
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = (torch.sigmoid(outputs) >= 0.5).int().cpu().numpy()
        return preds.flatten()

    def save_model(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.network[0].in_features
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_model(cls, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=True)
        instance = cls()
        instance.model = LivenessNet(input_size=checkpoint['input_size']).to(instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()
        return instance