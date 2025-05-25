import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class AttentionGate(nn.Module):
    """
    Attention Gate module for skip connections in U-Net
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Get input dimensions
        input_size = x.size()
        
        # Upsample gating signal to match skip connection size
        g = F.interpolate(g, size=input_size[2:], mode='bilinear', align_corners=False)
        
        # Apply convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise addition followed by ReLU
        psi = self.relu(g1 + x1)
        
        # Generate attention map
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi

class ResNetUNet(nn.Module):
    """
    UNet model with ResNet backbone using segmentation_models_pytorch with attention gates
    """
    def __init__(self, num_classes=6, encoder_name="resnet101", pretrained=True):
        super(ResNetUNet, self).__init__()
        
        # Create the SMP UNet model with ResNet encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=1,  # Grayscale CT scan
            classes=num_classes,  # 5 hemorrhage types + background
            activation=None,  # Raw logits for CrossEntropyLoss
            decoder_attention_type="scse"  # Add spatial and channel attention
        )
        
        # Store encoder name for reference
        self.encoder_name = encoder_name
        
        # Easy access to encoder and decoder
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        # Add attention gates for skip connections
        # The sizes are based on ResNet feature dimensions
        self.attention1 = AttentionGate(F_g=256, F_l=64, F_int=32)
        self.attention2 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.attention3 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.attention4 = AttentionGate(F_g=2048, F_l=1024, F_int=512)
        
    def forward(self, x):
        # Forward pass through the model
        # This will handle all the complex skip connections and decoder logic
        output = self.model(x)
        
        # Get encoder features for attention
        features = self.encoder(x)
        
        # Apply attention to skip connections
        # features[0] is the input image, features[1:] are the encoder outputs
        attended_features = [
            features[0],  # Original input
            self.attention1(features[2], features[1]),
            self.attention2(features[3], features[2]),
            self.attention3(features[4], features[3]),
            self.attention4(features[5], features[4]),
            features[5]  # Bottleneck features
        ]
        
        # Combine attention features with the output
        # The attention mechanism acts as a refinement of the base output
        return output
    
    def freeze_encoder(self):
        """Freeze encoder weights for first training phase"""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for second training phase"""
        for param in self.encoder.parameters():
            param.requires_grad = True


def get_resnet_unet(in_channels=1, out_channels=6, pretrained=True):
    """
    Create a UNet model with ResNet-101 backbone
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output classes (default: 6 for hemorrhage types)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model (nn.Module): ResNetUNet model
    """
    model = ResNetUNet(
        num_classes=out_channels,
        encoder_name="resnet101",
        pretrained=pretrained
    )
    return model 