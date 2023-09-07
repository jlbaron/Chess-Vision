'''
Convolutional Neural Network
takes squares of the board as input
produces class scores for each square
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from utils import label_to_string
import numpy as np

# could convert this to take each square in sequence, like a 2 part network
# this way it is judged only on the most important part: classifying a square
# separate the process of creating 64 squares and classifying the squares
# sequence_maker: takes batchx400x400x3 and converts to batchx64x2500
# classifier: treat each item in sequence as a full "unit" in the forward pass
#       compute loss per image instead of over the whole sequence at once

class BoardSplicer(nn.Module):
    def __init__(self, channels=3, patch_size=50, num_patches=64, max_len=64, device='cpu', image_analysis=[0, 0, 0]):
        super(BoardSplicer, self).__init__()
        self.max_len = max_len
        self.device_name = device
        self.image_analysis = image_analysis
        self.patch_size = patch_size
        # TODO: add an assert to make sure patch size divides evenly
        self.sqrt_patch_size = int(math.sqrt(num_patches))
        self.num_patches = num_patches

        #400x400x3 -> 400x400x1
        reduced_channel_num = 1
        self.channel_reduce = nn.Conv2d(channels, reduced_channel_num, 1, 1, device=device) # size 1 conv
        # split into 64 images

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
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
    def forward(self, x, labels=None):
        x = self.normalize_image(x)
        x = self.channel_reduce(x)
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
        return x

class SquareClassifier(nn.Module):
    def __init__(self, square_size=50, vocab_dim=13, device='cpu', ):
        super(SquareClassifier, self).__init__()
        self.device_name = device
        self.square_size = square_size
        self.linear = nn.Linear(square_size*square_size, vocab_dim, device=device)
    def forward(self, x):
        x = self.linear(x)
        return x
    
class ImageCNN(nn.Module):
    def __init__(self, channels=3, patch_size=50, num_patches=64, max_len=64, vocab_dim=13, device='cpu', image_analysis=[0, 0, 0]):
        super(ImageCNN, self).__init__()
        self.max_len = max_len
        self.device_name = device
        self.image_analysis = image_analysis
        self.patch_size = patch_size
        # TODO: add an assert to make sure patch size divides evenly
        self.sqrt_patch_size = int(math.sqrt(num_patches))
        self.num_patches = num_patches

        #400x400x3 -> 400x400x1
        reduced_channel_num = 1
        self.channel_reduce = nn.Conv2d(channels, reduced_channel_num, 1, 1, device=device) # size 1 conv
        # split into 64 images
        # downsample each image?: 50-10+1 = 41
        # self.dim_reduce = nn.Conv2d(1, 1, 5, 1, device=device)
        # classify each image
        self.linear = nn.Linear(patch_size*patch_size, vocab_dim, device=device)
        self.softmax = nn.Softmax(dim=-1)

    def normalize_image(self, image_batch):
            image_batch = image_batch.float() / 255.0
            return image_batch
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
    def forward(self, x, labels=None):
        x = self.normalize_image(x)
        x = self.channel_reduce(x)
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
        b, s, _ = x.shape
        # Flatten the batch and sequence dimensions
        x = x.view(-1, x.size(-1))
        # Pass the flattened input through the linear layer
        x = self.linear(x)
        # Reshape the output back to the original shape
        x = x.view(b, s, -1)
        # softmax and argmax for class scores
        x = self.softmax(x)
        x = torch.argmax(x, dim=2)
        return x