import yaml
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.CNN import BoardSplicer, SquareClassifier
from utils import ChessboardDataset, label_to_string

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_CNN.yaml', help='Path to the configuration file. Default: .\\configs\\config_CNN.yaml')
parser.add_argument('--sample', default='data\\test\\1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg', help='Path to the sample image file. Default: data\\test\\1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg')
parser.add_argument('--eval-test', default=False, help='Specify whether to evaluate the test or not. Default: False')

# load model from path
def load_model(model_path, splicer_path):
    model = SquareClassifier()  # Replace with your actual model class
    model.load_state_dict(torch.load(model_path))
    model.eval()

    splicer = BoardSplicer(image_analysis=args.image_analysis)
    splicer.load_state_dict(torch.load(splicer_path))
    splicer.eval()
    return model, splicer

# run through eval set and save results to a csv called test_results.csv
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

    output.to_csv("visualizations\\test_results.csv")

# create a label from a single sample
# currently only supports the size I trained on (400x400x3)
def process_sample(model, splicer, sample):
    #TODO: resize image to 400x400x3 if not already
    image = splicer(sample.unsqueeze(0))
    image_shape = image.shape
    image = image.squeeze(0)
    prediction = model(image)
    prediction = torch.argmax(torch.softmax(prediction, dim=-1), dim=-1)
    prediction = prediction.reshape(image_shape[0], image_shape[1])
    formatted = label_to_string(prediction[0])
    raw = ''.join([str(int(i)) for i in prediction[0]])
    return raw, formatted

'''
Usage: python inference.py [OPTIONS]

Options:
  --config CONFIG_PATH  Path to the configuration file.
                        Default: .\\configs\\config_CNN.yaml

  --sample SAMPLE_PATH  Path to the sample image file.
                        Default: data\\test\\1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg

  --eval-test EVAL_TEST Specify whether to evaluate the test or not.
                        Default: False
'''
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

    # compare predictions to truth and output as a full csv
    if args.eval_test:
        # Load the data loaders
        test_dataset = ChessboardDataset('data/test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        output_test(model, splicer, test_loader)

    # generate label on novel sample
    sample_path = args.sample
    assert(sample_path.endswith('.jpeg'))
    sample = torch.io.read_image(sample_path)
    raw, formatted = process_sample(model, splicer, sample)
    print(raw, formatted)

if __name__ == "__main__":
    main()