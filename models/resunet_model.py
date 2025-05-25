import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ResNetUNet(nn.Module):
    """
    UNet model with ResNet backbone using segmentation_models_pytorch
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
        )
        
        # Store encoder name for reference
        self.encoder_name = encoder_name
        
        # Easy access to encoder and decoder
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)
    
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