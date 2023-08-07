'''
Script to load data and train model
'''

import yaml
import argparse
import torch
import numpy as np
import torch.nn as nn
from os import listdir
from torch.utils.data import DataLoader
from models.VisionTransformer import ImageTransformer
from utils import ChessboardDataset, calculate_accuracy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='./config.yaml')

if __name__ ==  '__main__':
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device type: {device}')

    # data
    train_data_dir = '/data/train'
    test_data_dir = '/data/test'

    train_dataset = ChessboardDataset(data_dir=train_data_dir)
    test_dataset = ChessboardDataset(data_dir=test_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # model
    if args.model == "VisionTransformer":
        model = ImageTransformer(d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.dim_feedforward, dropout=args.dropout, num_layers=args.num_layers, num_classes=args.num_classes, ispool=args.ispool, device=device).to(device)
    else:
        raise Exception("Invalid model")

    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss")


    # source used as reference: https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(args.epochs):
        # training loop
        # source used as reference: assignment 2 boilerplate
        batch_losses = []
        batch_accuracies = []
        model.train()
        for idx, (images, labels) in enumerate(train_loader):
            # put on device
            images = videos.to(device)
            labels = labels.to(device)

            # perform forward pass
            optimizer.zero_grad()
            predictions = model(videos)

            # backward pass
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            # calculate accuracy

            batch_losses.append(loss.item())
            batch_accuracies.append(batch_accuracy)


        avg_train_loss = sum(batch_losses) / len(batch_losses)
        avg_train_acc = sum(batch_accuracies) / len(batch_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        # validation loop
        # source used as reference: assignment 2 boilerplate
        val_batch_losses = []
        val_batch_accuracies = []
        all_predictions = None
        all_labels = None
        model.eval()
        with torch.no_grad():
            for idx, (videos, labels) in enumerate(test_loader):
                # enable gpu
                videos = videos.to(device)
                labels = labels.to(device)

                # perform forward pass
                predictions = model(videos)
                loss = criterion(predictions, labels)
                
                
                # calculate accuracy
                batch_accuracy = calculate_accuracy(predictions, labels)


                # append to loop variables
                val_batch_losses.append(loss.item())
                val_batch_accuracies.append(batch_accuracy)
                all_predictions = predictions if all_predictions is None else torch.cat((all_predictions, predictions), 0)
                all_labels = labels if all_labels is None else torch.cat((all_labels, labels), 0)


        avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
        avg_val_acc = sum(val_batch_accuracies) / len(val_batch_accuracies)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f"--------EPOCH {epoch+1}, TRAIN LOSS: {avg_train_loss}, TRAIN ACC: {avg_train_acc}, VAL LOSS: {avg_val_loss}, VAL ACC: {avg_val_acc}---------")
        print("---------------------------------------------------")

    epochs_range = range(1, args.epochs+1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    plt.show()
    plt.plot(train_accuracies, label="Training Accuracies")
    plt.plot(val_accuracies, label="Validation Accuracies")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("acc.png")
    plt.show()