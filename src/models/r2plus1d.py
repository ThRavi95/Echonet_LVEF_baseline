import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from typing import Optional


class EchoNetR2Plus1D(nn.Module):
    """
    Spatiotemporal R(2+1)D-18 Network adapted for continuous LVEF regression.
    
    Factorizes 3D convolutions into separate spatial (1x3x3) and temporal (3x1x1)
    operations. Pretrained on Kinetics-400 and mapped to a scalar regression output.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load backbone with Kinetics-400 weights
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)

        # Freeze earlier feature layers if linear probing is requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # In R(2+1)D-18, the feature dimension before fc is 512
        in_features = self.backbone.fc.in_features
        
        # Replace 400-class action classification head with scalar regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input video tensor of shape (Batch, 3, 32, 112, 112).
            
        Returns:
            Scalar predictions of shape (Batch,) representing predicted LVEF %.
        """
        # (B, 1) -> (B,)
        out = self.backbone(x)
        return out.squeeze(-1)

    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the final convolutional block of layer4.
        Used by 3D Grad-CAM for gradient hook registration.
        """
        # layer4[-1] is the last BasicBlock in R(2+1)D
        return self.backbone.layer4[-1]
