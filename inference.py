'''
Script to perform various analyses on trained model
Currently implemented confusion matrix and working on t-SNE
reference classes:
    'CH': 0 : Changeup
    'CU': 1 : Curveball
    'EP': 2 : Eephus
    'FC': 3 : Cutter
    'FF': 4 : Four-seam Fastball
    'FS': 5 : Splitter
    'FT': 6 : Two-seam Fastball
    'KC': 7 : Knuckle Curve
    'KN': 8 : Knuckleball
    'SC': 9 : Screwball
    'SI': 10 : Sinker
    'SL': 11 : Slider
'''
import yaml
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.CNN import BoardSplicer, SquareClassifier
from utils import ChessboardDataset, label_to_string

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_CNN.yaml')

def load_model(model_path, splicer_path):
    model = SquareClassifier()  # Replace with your actual model class
    model.load_state_dict(torch.load(model_path))
    model.eval()

    splicer = BoardSplicer(image_analysis=args.image_analysis)
    splicer.load_state_dict(torch.load(splicer_path))
    splicer.eval()
    return model, splicer

def output_test(model, splicer, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    splicer.to(device)

    y_true_raw = []
    y_pred_raw = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = splicer(images, labels)
            image_shape = images.shape
            labels = labels.flatten(0)
            images = images.flatten(0, 1)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            # perform forward pass
            predictions = model(images)
            predictions = torch.argmax(torch.softmax(predictions, dim=-1), dim=-1)
            predictions = predictions.reshape(image_shape[0], image_shape[1])
            labels = labels.reshape(image_shape[0], image_shape[1])

            for i in range(image_shape[0]):
                y_true_raw.append(''.join([str(int(i)) for i in labels[i]]))
                y_pred_raw.append(''.join([str(int(i)) for i in predictions[i]]))
                y_true.append(label_to_string(labels[i]))
                y_pred.append(label_to_string(predictions[i]))

    # Generate classification report
    output = {'y_pred': y_pred, 'y_true': y_true, 'y_pred_raw' : y_pred_raw, 'y_true_raw' : y_true_raw}
    output = pd.DataFrame(output)
    print(output.head())

    output.to_csv("data\\test_results.csv")

def main():
    #args from yaml file
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # set args object
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Load the model
    model_path = 'checkpoints/split_CNN_trained.pt'
    splicer_path = 'checkpoints/splicer_trained.pt'
    model, splicer = load_model(model_path, splicer_path)

    # Load the data loaders
    test_dataset = ChessboardDataset('data/test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # compare predictions to truth
    output_test(model, splicer, test_loader)

if __name__ == "__main__":
    main()