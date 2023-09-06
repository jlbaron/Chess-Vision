'''
data uses Forsyth-Edwards Notation with dashes instead of slashes
this notation comes with 6 parts
1) pieces by rank (separated with -)
2) active color
3) castling availability
4) en passant target square
5) halfmove clock
6) fullmove number

to make the data easier to classify, I only include the pieces by rank
for example here is the opening (uppercase is white): 
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
and after e4:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR

Will only use 1) pieces by rank since other parts are nearly impossible to determine
'''
import os
import numpy as np
import re
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

# - is a special character that wont be learned
vocab = {
    "-": 999,
    "0": 0,
    "p": 1, "P": 2,
    "n": 3, "N": 4,
    "b": 5, "B": 6,
    "r": 7, "R": 8,
    "q": 9, "Q": 10,
    "k": 11, "K": 12
}
dash_idx = 999

inv_vocab = {v: k for k, v in vocab.items()}


class ChessboardDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_list = [filename for filename in os.listdir(data_dir) if filename.endswith('.jpeg')]
        self.max_len = 71 # 8x8=64 + 7 dashes

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # This process needs to change
        # Retrieving the image is bugged and the label is likely incorrect
        # Although the label tools were given with the dataset, I would like to try just a straight ce loss on the label as is
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        image = torchvision.io.read_image(img_name)
        label = self.image_list[idx].split('.')[0]
        expanded = ""
        for i in label:
            if i.isdigit():
                for j in range(int(i)):
                    expanded += "0"
            else:
                expanded += i
        label = [vocab[char] for char in expanded]  # Convert characters to vocab values
        label = [i for i in label if i != dash_idx]
        label = torch.tensor(label, dtype=torch.int32)
        return image.to(torch.long), label
    
'''
Note on labels:
    In FEN spaces are grouped so 3 blank squares yields a 3 instead of 3 0s
    This could easily be taken care of outside of the network
    Essentially encourage the network to only give 0s for every blank square
    Then convert labels to and from proper notation
    This has a few benefits:
        predictable max_len
        easier for transformer to see 64 squares and output 64 values
'''
# convert label that has been formatted for learning back to original form
# data will always be 64 tokens and needs to be expanded to 71 tokens where every 9th token is dash
# additionally the final string needs to have 0s compressed
def label_to_string(label):
    label_as_list = label.tolist()
    j = 0
    while j < len(label_as_list):
        if (j + 1) % 9 == 0:
            label_as_list.insert(j, dash_idx)  # Insert dash_idx at the current index j
        j += 1
    counter = 0
    new_label = ""
    for i in label_as_list:
        if i == 0:
            counter += 1
        else:
            if counter > 0:
                new_label += str(counter)
                new_label += inv_vocab[i]
                counter = 0
            else:
                new_label += inv_vocab[i]
    if counter > 0:
        new_label += str(counter)
    return new_label

def calculate_accuracy(predictions, labels):
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / labels.shape[1]
    return accuracy

# for every token in predictions count if the same as token in labels and div by total
# 71 different class scores for 13 classes (ie 71 probabilities of 13 tokens = 71 tokens)
def per_word_acc(predictions, labels):
    accuracy = (predictions == labels).float().mean(dim=1)
    return accuracy.mean().item()

def test_data_processing(data_dir, start, stop):
    image_list = [filename for filename in os.listdir(data_dir) if filename.endswith('.jpeg')]
    max_len = max(len(filename.split('.')[0]) for filename in image_list)
    for i in range(start, stop):
        # img_name = os.path.join(data_dir, image_list[i])
        # image = torchvision.io.read_image(img_name)
        label = image_list[i].split('.')[0]
        expanded = ""
        for i in label:
            if i.isdigit():
                for j in range(int(i)):
                    expanded += "0"
            else:
                expanded += i
        print(expanded)
        label = [vocab[char] for char in expanded]  # Convert characters to ASCII values
        print(label)
        label = torch.tensor(label, dtype=torch.int32)
        # image = torch.LongTensor(image)
        # print(image.shape, image.dtype)
        # print(label, label.dtype)

# test_data_processing('data/train', 128*37, 128*39)