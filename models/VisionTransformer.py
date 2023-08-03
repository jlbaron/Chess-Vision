'''
setting up main loop
'''
import torch
import torch.nn as nn


class ImagePatchEncoder(nn.Module):
    def __init__(self, d_model, patches=50, channels=3, device='cpu'):
        super(ImagePatchEncoder, self).__init__()
        self.patches = patches

        #conv
        self.conv = nn.Conv2d(channels, d_model, patches, stride=patches, device=device)

        #pos encoding

    def forward(self, x):
        #board is 8x8 image is 400x400
        #stride a 50x50xd_model kernel with stride 50 = 8x8xd_model
        x = self.conv(x)
        print(x.shape)
        #flatten to tokens of shape 64xd_model
        x = x.flatten(0)

        #add positional encoding

        return x
    
class ImageTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=1, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super(ImageTransformer, self).__init__()

        self.embedding = ImagePatchEncoder(d_model=d_model, device=device)

        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token_emb = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
    def forward(self, x):
        x = self.normalize_image(x)
        x = self.embedding(x)
        x = self.encoder(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x]) #prepend CLS
        x = x[0] #extract CLS
        return x