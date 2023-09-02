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
import os

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_VisionTransformer.yaml')

def train(model, train_loader, optimizer, criterion, device):
    total_loss = 0.
    total_acc = 0.
    model.train()
    for idx, (images, labels) in enumerate(train_loader):
        # put on device
        images = images.to(device)
        labels = labels.to(device)

        # perform forward pass
        optimizer.zero_grad()
        predictions = model(images, labels)

        # backward pass
        loss = criterion(predictions, labels.to(torch.float))
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        total_loss += loss.item()
        acc = calculate_accuracy(predictions, labels)
        total_acc += acc
        print(f"Batch: {idx}, Loss: {loss.item()}, Acc: {acc}")
    return model, total_loss / len(train_loader), total_acc / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.
    total_acc = 0.
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            # enable gpu
            images = images.to(device)
            labels = labels.to(device)

            # perform forward pass
            predictions = model(images)
            loss = criterion(predictions, labels.to(torch.float))
            
            total_loss += loss.item()
            acc = calculate_accuracy(predictions, labels)
            total_acc += acc
            print(f"Batch: {idx}, Loss: {loss.item()}, Acc: {acc}")
    return total_loss / len(val_loader), total_acc / (len(val_loader))

def plot_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("visualizations\\training_plots\\loss.png")
    plt.show()
    plt.plot(train_accuracies, label="Training Accuracies")
    plt.plot(val_accuracies, label="Validation Accuracies")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("visualizations\\training_plots\\acc.png")
    plt.show()

if __name__ ==  '__main__':
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device type: {device}')

    # data
    train_data_dir = 'data/train'
    val_data_dir = 'data/test'

    train_dataset = ChessboardDataset(data_dir=train_data_dir)
    val_dataset = ChessboardDataset(data_dir=val_data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # model
    if args.model == "VisionTransformer":
        model = ImageTransformer(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout, max_len=train_dataset.max_len, device=device, image_analysis=args.image_analysis).to(device)
    else:
        raise Exception("Invalid model")

    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss")


    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(args.epochs):
        model, train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"--------EPOCH {epoch+1}, TRAIN LOSS: {train_loss}, TRAIN ACC: {train_acc}, VAL LOSS: {val_loss}, VAL ACC: {val_acc}---------")
        print("---------------------------------------------------")
    model_path = os.path.join('checkpoints', f"{args.model}_trained.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to '{model_path}'.")
    plot_curves(train_losses, train_accuracies, val_losses, val_accuracies)