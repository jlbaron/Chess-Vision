'''
setting up main loop
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageSquareEncoder(nn.Module):
    def __init__(self, d_model, patch_size=50, num_patches=64, channels=3, device='cpu'):
        super(ImageSquareEncoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        # embed board spaces into d_model
        self.embedding = nn.Linear(self.patch_size * self.patch_size * channels, d_model)

        # Create positional encoding for each patch
        self.positional_encoding = self.create_positional_encoding(d_model, self.num_patches, device)

    def create_positional_encoding(self, d_model, num_patches, device='cpu'):
        # sine positional encoding

        positional_encoding = torch.zeros(num_patches, d_model).to(device)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding

    def forward(self, x):
        # Input x: (batch_size, height, width, channels)

        batch_size, height, width, channels = x.size()

        # Reshape the input image to separate patches
        # unfold on dim 1 (height) and dim 2 (width) with patch size
        x = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        # reform and use contiguous for memory layout
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * channels)

        # Downsize each 50x50 patch to 16x16
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)

        # Pass through the embedding layer
        x = self.embedding(x)

        # Add positional encoding to each patch
        x = x + self.positional_encoding

        return x
    
class ImagePatchEncoder(nn.Module):
    def __init__(self, d_model, patch_size=20, num_patches=400, channels=3, device='cpu'):
        super(ImagePatchEncoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 400-20= 380/20 = 19+1 = 20
        self.cnn = nn.Conv2d(channels, d_model, patch_size, patch_size)

        # Create positional encoding for each patch
        self.positional_encoding = self.create_positional_encoding(d_model, self.num_patches, device)

    def create_positional_encoding(self, d_model, num_patches, device='cpu'):
        # sine positional encoding

        positional_encoding = torch.zeros(num_patches, d_model).to(device)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding

    def forward(self, x):
        # Input x: (batch_size, height, width, channels)

        # use a cnn to reduce x to d_modelx20x20
        x = self.cnn(x)
        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        # Add positional encoding to each patch
        x = x + self.positional_encoding

        return x
    
class ImageTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=1, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super(ImageTransformer, self).__init__()

        self.embedding = ImagePatchEncoder(d_model=d_model, device=device)

        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token_emb = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
    def forward(self, x):
        x = self.normalize_image(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x]) #prepend CLS
        x = x[0] #extract CLS
        return x