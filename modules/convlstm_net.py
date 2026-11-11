import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BEHAVIOR_CLASSES: List[str] = [
    "NORMAL_WALKING",
    "RUNNING_FLEEING",
    "THEFT_SNATCHING_ROBBERY",
    "PHYSICAL_STRUGGLE_FIGHT",
    "SUDDEN_FALL_COLLAPSE",
    "LOITERING_LINGERING",
    "RESTRICTED_INTRUSION",
    "ERRATIC_PACING",
    "STATIONARY_IDLE",
    "SCENE_EMPTY_SECURE",
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(BEHAVIOR_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(BEHAVIOR_CLASSES)}


class ConvLSTMCell(nn.Module):
    """
    2D Spatiotemporal Convolutional Long Short-Term Memory (ConvLSTM) Cell.
    Preserves 2D spatial features across recurrent time steps.
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (Batch, in_channels, H, W)
        state: (h, c) each (Batch, hidden_channels, H, W)
        """
        batch_size, _, height, width = x.size()
        if state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, c_tilde, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_tilde = torch.tanh(c_tilde)
        o = torch.sigmoid(o)

        c_next = f * c + i * c_tilde
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMBehaviorClassifier(nn.Module):
    """
    End-to-End Deep Spatiotemporal Neural Network for Human Behavior Classification.
    Accepts sequence of spatial tensors (Batch, Time, Channels, Height, Width).
    Extracts 2D spatiotemporal dynamics via CNN-ConvLSTM recurrence, and classifies
    behaviors via neural Softmax head.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = len(BEHAVIOR_CLASSES),
        cnn_channels: int = 24,
        lstm_hidden: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden

        # 1. 2D Spatial Feature Encoder per time-step
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 2. Spatiotemporal ConvLSTM Recurrent Layer
        self.convlstm = ConvLSTMCell(
            in_channels=cnn_channels,
            hidden_channels=lstm_hidden,
            kernel_size=3,
        )

        # 3. Spatial Pooling Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Neural Classification MLP Head (Concatenates final state + temporal sequence average)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: (Batch, Time, Channels, Height, Width)
        Returns:
            logits: (Batch, num_classes)
            last_h: (Batch, lstm_hidden, H_feat, W_feat)
        """
        batch_size, seq_len, c, h, w = x_seq.size()
        state = None
        h_list = []

        for t in range(seq_len):
            xt = x_seq[:, t]  # (B, C, H, W)
            feat_t = self.spatial_encoder(xt)  # (B, cnn_channels, H/2, W/2)
            h_t, c_t = self.convlstm(feat_t, state)
            state = (h_t, c_t)
            h_list.append(h_t)

        final_h = state[0]
        h_stack = torch.stack(h_list, dim=1)  # (B, T, hidden, H/2, W/2)
        pooled_seq = self.global_pool(
            h_stack.view(batch_size * seq_len, self.lstm_hidden, h // 2, w // 2)
        ).view(batch_size, seq_len, self.lstm_hidden)

        temporal_mean = torch.mean(pooled_seq, dim=1)  # (B, hidden)
        final_pooled = pooled_seq[:, -1]  # (B, hidden)
        rep = torch.cat([final_pooled, temporal_mean], dim=1)  # (B, 2 * hidden)
        logits = self.classifier(rep)
        return logits, final_h
        return logits, final_h

    @torch.no_grad()
    def predict_sequence(
        self,
        x_np: np.ndarray,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Accepts numpy array of shape (T, C, H, W) or (1, T, C, H, W).
        Returns predicted class name, confidence, and probability map.
        """
        self.eval()
        if x_np.ndim == 4:
            x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)  # (1, T, C, H, W)
        else:
            x_t = torch.from_numpy(x_np).to(device)

        logits, final_h = self.forward(x_t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = IDX_TO_CLASS[pred_idx]
        confidence = float(probs[pred_idx])

        # Extract recurrent feature heatmap for HUD / PIP visualization
        feat_map = final_h.squeeze(0).mean(dim=0).cpu().numpy()

        all_probs = {IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(len(probs))}

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": all_probs,
            "feature_heatmap": feat_map,
        }
