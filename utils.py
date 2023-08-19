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

I am not sure if any model can look at a random board and say:
1) what the last move was
2) how many moves have been played
both of which are needed for 2-6 in the full FEN
'''
import os
import numpy as np
import re
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class ChessboardDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_list = [filename for filename in os.listdir(data_dir) if filename.endswith('.jpeg')]
        self.max_len = max(len(filename.split('.')[0]) for filename in self.image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # This process needs to change
        # Retrieving the image is bugged and the label is likely incorrect
        # Although the label tools were given with the dataset, I would like to try just a straight ce loss on the label as is
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        image = torchvision.io.read_image(img_name) #idk i need the internet
        label = self.image_list[idx].split('.')[0]
        label = [ord(char) for char in label]  # Convert characters to ASCII values
        label = torch.tensor(label, dtype=torch.int32)
        pad_value = 0
        label = torch.cat([label, torch.tensor([pad_value] * (self.max_len - len(label)))])
        return image, label

#rely on https://www.kaggle.com/code/koryakinp/chess-fen-generator/notebook
#rely on https://www.kaggle.com/code/ananyaroy1011/chess-positions-fen-prediction-eda-cnn-model
piece_symbols = 'prbnkqPRBNKQ'
def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

# convert fen to number for classification
def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if(char in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)

    return output


# convert transformer output to fen to pretty printing
def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

def calculate_accuracy(predictions, labels):
    _, predicted_classes = torch.max(predictions, 1)
    correct_predictions = (predicted_classes == labels).sum().item()
    total_predictions = labels.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy