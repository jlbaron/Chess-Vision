'''
setting up main loop
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
    
class ImagePatchEncoder(nn.Module):
    def __init__(self, d_model, patch_size=20, num_patches=400, channels=3, device='cpu', image_analysis=False):
        super(ImagePatchEncoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 400-20= 380/20 = 19+1 = 20
        self.cnn = nn.Conv2d(channels, d_model, patch_size, patch_size)

        #TODO: after consulting with visualizations I see the need for some changes
        # split image into 8x8 images then pass each through a conv, size 1, to get 8x8x1
        # embed to d_model so that there are 64xd_model patches of the board
        # positional encoding as normal

        # Create positional encoding for each patch
        self.positional_encoding = self.create_positional_encoding(d_model, self.num_patches, device)

        self.image_analysis = image_analysis
        

    def create_positional_encoding(self, d_model, num_patches, device='cpu'):
        # sine positional encoding

        positional_encoding = torch.zeros(num_patches, d_model).to(device)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding

    def visualize_cnn_output(self, cnn_output, filename):
        plt.imshow(cnn_output[0].cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
        plt.axis('off')
        plt.savefig(filename)  # Save the figure with the provided filename

    def forward(self, x, labels=None):
        # Input x: (batch_size, height, width, channels)

        # use a cnn to reduce x to d_modelx20x20
        x = self.cnn(x)

        if self.image_analysis and labels is not None:
            # Select a random sample index from the batch
            random_sample = torch.randint(0, x.shape[0], (1,)).item()
            
            # use label but need to convert back to string (it is an ASCII list)
            header = "visualizations\\cnn_output\\"
            label = ""
            for val in labels[random_sample]:
                label += chr(val)
            # label = "visualizations\\cnn_output\\5NR1-4P3-1P6-N4N2-8-4BB2-7k-1K5Q"
            filename = header+ label.replace(chr(0), "") + ".png"
            
            # Create and save the CNN output visualization
            # self.visualize_cnn_output(x[random_sample], str(filename))
            self.visualize_cnn_output(x[random_sample], filename)


        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        # Add positional encoding to each patch
        x = x + self.positional_encoding

        return x
    
class ImageTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=1, dim_feedforward=2048, dropout=0.1, max_len=45, device='cpu', image_analysis=False):
        super(ImageTransformer, self).__init__()

        self.embedding = ImagePatchEncoder(d_model=d_model, device=device, image_analysis=image_analysis)

        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token_emb = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=False)
        self.linear = nn.Linear(d_model, max_len)

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
    def forward(self, x, labels=None):
        x = self.normalize_image(x)
        x = self.embedding(x, labels)
        x = self.transformer_encoder(x)
        cls_token_emb = self.cls_token_emb.expand(x.shape[0], 1, -1)
        x = torch.cat([cls_token_emb, x], dim=1) #prepend CLS
        x = x[:, 0, :] #extract CLS
        x = self.linear(x)
        return x