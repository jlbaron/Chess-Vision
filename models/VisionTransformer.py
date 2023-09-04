'''
setting up main loop
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
feed through a convolution layer for dimensionality reduction
do a kernel 1 convolution with d_model output features to get 64x50x50xd_model (drop 50x50 to get 64xd_model)
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
        split_image = torch.zeros([x.shape[0], self.num_patches, self.patch_size, self.patch_size], device=self.device_name)
        # needs to iterate as row/col
        iter = 0
        for h in range(self.sqrt_patch_size):
            for w in range(self.sqrt_patch_size):
                h_start = h*self.patch_size
                h_end = h_start+self.patch_size
                w_start = w*self.patch_size
                w_end = w_start+self.patch_size
                split_image[:, iter] = x[:, h_start:h_end, w_start:w_end]
                iter += 1

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
    def __init__(self, d_model=512, nhead=4, num_layers=1, dim_feedforward=2048, dropout=0.1, max_len=71, vocab_dim=13, device='cpu', image_analysis=[0, 0, 0]):
        super(ImageTransformer, self).__init__()
        self.max_len = max_len
        self.device_name = device
        self.embedding = ImagePatchEncoder(d_model=d_model, device=device, image_analysis=image_analysis)

        encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, device=device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.linear = nn.Linear(d_model, max_len)
        self.output_layer = nn.Linear(d_model, vocab_dim)  # Single linear layer for all positions

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
    # TODO: incomplete, this is meant to take [batch, seq] where seq=64 and append 7 "-" at the correct idx to make seq==maxlen (71)
    def format_output_sequence(self, x):
        # for every batch: append a [999] in idx 7, 15, 23, 31, 39, 47, 55
        dash_idx = torch.tensor([999], device=self.device_name) # idx of - in vocab
        expanded_x = torch.empty([x.shape[0], self.max_len], device=self.device_name, requires_grad=False) # [batch, max_len]
        # loop through x, tracking when to also append dash
        for b in range(x.shape[0]):
            j = 0
            for i in range(x.shape[1]):
                if (j+1) % 9 == 0:
                    expanded_x[b][j] = dash_idx
                    j += 1
                expanded_x[b][j] = x[b][i]
                j += 1
        return expanded_x
    def forward(self, x, labels=None):
        x = self.normalize_image(x)
        x = self.embedding(x, labels)
        x = self.transformer_encoder(x)
        # convert to sequence of class probabilities, softmax, then argmax to get 64 tokens
        x = torch.softmax(self.output_layer(x), dim=-1)
        x = torch.argmax(x, dim=2)
        # x = self.format_output_sequence(x)
        return x