'''
Vision transformer
takes embedded board squares as input
produces sequence of class guesses for pieces
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from utils import label_to_string
import numpy as np

'''
400x400x3
convolution to turn into single channel
400x400x1
split into 64 sections of 50x50
64x50x50
flatten and linear project into embeddings
output 64x512
'''
class ImagePatchEncoder(nn.Module):
    def __init__(self, d_model, patch_size=50, num_patches=64, channels=3, device='cpu', image_analysis=[0, 0, 0]):
        super(ImagePatchEncoder, self).__init__()
        self.patch_size = patch_size
        # TODO: add an assert to make sure patch size divides evenly
        self.sqrt_patch_size = int(math.sqrt(num_patches))
        self.num_patches = num_patches
        self.device_name = device

        #400x400x3 -> 400x400x1
        self.cnn = nn.Conv2d(channels, 1, 1, 1, device=device)

        # (unsqueeze final index) -> 400x400
        # split into 64x50x50, flatten to 64x2500
        # linear layer to embed 2500 to d_model -> 64xd_model
        self.linear = nn.Linear(patch_size*patch_size, d_model, device=device)

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
        seq = cnn_output.shape[0]
        wspace = 0.1
        hspace = 0.1
        fig, axes = plt.subplots(self.sqrt_patch_size, self.sqrt_patch_size, figsize=(12, 12))
        axes = axes.flatten()
        for s in range(seq):
            # Extract the current patch
            patch = cnn_output[s].cpu().detach().numpy()

            # Display the patch in the corresponding subplot
            axes[s].imshow(patch, cmap='viridis', interpolation='nearest')
            axes[s].axis('off')
        plt.imshow(cnn_output[0].cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(filename)  # Save the figure with the provided filename
    
    def visualize_embeddings(self, embeddings, filename, is_heatmap):
        seq = embeddings.shape[0]
        if is_heatmap:
            embedding_matrix = embeddings.cpu().detach().numpy().T
            plt.figure(figsize=(12, 6))
            plt.imshow(embedding_matrix, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('HeatMap of '+filename[:-4])
            plt.xlabel('Patch')
            plt.ylabel('Embedding Dim')
            plt.savefig(filename)
        else:
            wspace = 0.1
            hspace = 0.1
            fig, axes = plt.subplots(self.sqrt_patch_size, self.sqrt_patch_size, figsize=(12, 12))
            axes = axes.flatten()
            for s in range(seq):
                # Extract the current patch
                patch = embeddings[s].cpu().detach().numpy()

                # Display the patch in the corresponding subplot
                axes[s].plot(patch)
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(filename)  # Save the figure with the provided filename

    def forward(self, x, labels=None):
        # Input x: (batch_size, height, width, channels)

        # use a cnn to compress color channels
        x = self.cnn(x)
        x = x.squeeze(1) # do away with the 1 in bx1x400x400

        # Reshape the input tensor `x` into a tensor of segments
        x_reshaped = x.view(x.size(0), self.sqrt_patch_size, self.patch_size, self.sqrt_patch_size, self.patch_size)

        # Transpose the dimensions to bring the segments together
        x_transposed = x_reshaped.permute(0, 1, 3, 2, 4)

        # Reshape the tensor to combine rows and columns into a single sequence of 64
        split_image = x_transposed.contiguous().view(x.size(0), self.num_patches, self.patch_size, self.patch_size)

        random_sample = None
        if self.image_analysis[0] and labels is not None:
            # Select a random sample index from the batch
            random_sample = torch.randint(0, split_image.shape[0], (1,)).item()
            
            header = "visualizations\\cnn_output\\"
            label = label_to_string(labels[random_sample])
            filename = header + label + ".png"
            
            self.visualize_cnn_output(split_image[random_sample], filename)

        x = split_image.flatten(2)
        x = self.linear(x)

        if self.image_analysis[1] and labels is not None:
            header1 = "visualizations\\projection_output\\"
            header2 = "visualizations\\projection_heat\\"
            label = label_to_string(labels[random_sample])
            filename1 = header1 + label + ".png"
            filename2 = header2 + label + ".png"
            
            self.visualize_embeddings(x[random_sample], filename1, is_heatmap=False)
            self.visualize_embeddings(x[random_sample], filename2, is_heatmap=True)

        # Add positional encoding to each patch
        x = x + self.positional_encoding

        if self.image_analysis[2] and labels is not None:
            header1 = "visualizations\\pe_output\\"
            header2 = "visualizations\\pe_heat\\"
            label = label_to_string(labels[random_sample])
            filename1 = header1 + label + ".png"
            filename2 = header2 + label + ".png"
            
            self.visualize_embeddings(x[random_sample], filename1, is_heatmap=False)
            self.visualize_embeddings(x[random_sample], filename2, is_heatmap=True)
        return x
    
class ImageTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=1, dim_feedforward=2048, dropout=0.1, max_len=64, vocab_dim=13, device='cpu', image_analysis=[0, 0, 0]):
        super(ImageTransformer, self).__init__()
        self.max_len = max_len
        self.device_name = device
        self.embedding = ImagePatchEncoder(d_model=d_model, device=device, image_analysis=image_analysis)

        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_dim)  # Single linear layer for all positions
        self.softmax = nn.Softmax(dim=-1)

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
    def forward(self, x, labels=None):
        x = self.normalize_image(x)
        x = self.embedding(x, labels)
        x = self.transformer_encoder(x)

        # convert to sequence of class probabilities, softmax, then argmax to get 64 tokens
        x = self.softmax(self.output_layer(x))
        x = torch.argmax(x, dim=2)
        # TODO: experiment with another technique to get tokens from transformer output
        # [batch, seq, d_model] -> [batch, seq, vocab_len]  -> [batch, seq]
        # or just [batch, seq, vocab_len]?
        return x