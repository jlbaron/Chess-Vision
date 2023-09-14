'''
Script to load data and train model
'''

import os
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ChessboardDataset, per_word_acc, inv_vocab
from models.VisionTransformer import ImageTransformer
from models.CNN import ImageCNN, BoardSplicer, SquareClassifier

parser = argparse.ArgumentParser(description='Chess-Vision')
parser.add_argument('--config', default='.\\configs\\config_CNN.yaml')
eval_labels = {"Pred" : [], "True" : []}

def train(model, splicer, train_loader, optimizer, criterion, device):
    is_cnn_split = True if args.model == "split_CNN" else False
    if is_cnn_split:
        splicer.eval()
    model.train()

    total_loss = 0.
    total_acc = 0.
    for idx, (images, labels) in enumerate(train_loader):
        # put on device
        images = images.to(device)
        labels = labels.to(device)

        image_shape = None
        if is_cnn_split:
            images = splicer(images, labels)
            image_shape = images.shape
            # convert labels [batch, seq] to one-hots [batch*seq]
            labels = labels.flatten(0)
            # convert images [batch, seq, token_len] to [batch*seq, token_len]
            images = images.flatten(0, 1)
            # images will become [batch*seq, vocab_dim] after forward pass
            labels = torch.tensor(labels, dtype=torch.long, device=device)

        # perform forward pass 
        optimizer.zero_grad()
        predictions = model(images)

        # backward pass
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        # calculate accuracy 
        if is_cnn_split:
            predictions = torch.argmax(torch.softmax(predictions, dim=-1), dim=-1)
            acc = per_word_acc(predictions.reshape(image_shape[0], image_shape[1]), labels.reshape(image_shape[0], image_shape[1]))
        else:
            acc = per_word_acc(predictions, labels)
        total_loss += loss.item()
        total_acc += acc
        print(f"Batch: {idx}, Loss: {loss.item()}, Acc: {acc}")
    return model, total_loss / len(train_loader), total_acc / len(train_loader)

def evaluate(model, splicer, val_loader, criterion, device):
    is_cnn_split = True if args.model == "split_CNN" else False
    if is_cnn_split:
        splicer.eval()
    model.eval()

    total_loss = 0.
    total_acc = 0.
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            # enable gpu
            images = images.to(device)
            labels = labels.to(device)

            image_shape = None
            if is_cnn_split:
                images = splicer(images, labels)
                image_shape = images.shape
                # convert labels [batch, seq] to one-hots [batch*seq]
                labels = labels.flatten(0)
                # convert images [batch, seq, token_len] to [batch*seq, token_len]
                images = images.flatten(0, 1)
                # images will become [batch*seq, vocab_dim] after forward pass
                labels = torch.tensor(labels, dtype=torch.long, device=device)

            # perform forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            # calculate accuracy
            if is_cnn_split:
                predictions = torch.argmax(torch.softmax(predictions, dim=-1), dim=-1)
                acc = per_word_acc(predictions.reshape(image_shape[0], image_shape[1]), labels.reshape(image_shape[0], image_shape[1]))
            else:
                acc = per_word_acc(predictions, labels)
            total_loss += loss.item()
            total_acc += acc

            if idx == 0:
                predictions = predictions.reshape(image_shape[0], image_shape[1])
                labels = labels.reshape(image_shape[0], image_shape[1])
                predicted_output = ''.join([inv_vocab[i.item()]+" " for  i in predictions[0]])
                label_output = ''.join([inv_vocab[i.item()]+" " for i in labels[0]])
                eval_labels["Pred"].append(predicted_output)
                eval_labels["True"].append(label_output)
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
    elif args.model == "CNN":
        model = ImageCNN(dropout=args.dropout, device=device, image_analysis=args.image_analysis)
    elif args.model == "split_CNN":
        splicer = BoardSplicer(device=device, image_analysis=args.image_analysis)
        model = SquareClassifier(device=device)
    else:
        raise Exception("Invalid model")

    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.8, 0.999))

    if args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss")


    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(args.epochs):
        # train and eval
        model, train_loss, train_acc = train(model, splicer, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, splicer, val_loader, criterion, device)

        # append to counters
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # save models
        model_path = os.path.join('checkpoints', f"{args.model}_trained.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to '{model_path}'.")
        if args.model == "split_CNN":
            model_path = os.path.join('checkpoints', f"splicer_trained.pt")
            torch.save(splicer.state_dict(), model_path)
            print(f"Splicer saved to '{model_path}'.")

        # print training info
        print(f"--------EPOCH {epoch+1}, TRAIN LOSS: {train_loss}, TRAIN ACC: {train_acc}, VAL LOSS: {val_loss}, VAL ACC: {val_acc}---------")
        print("---------------------------------------------------")
    # save eval_labels
    eval_df = pd.DataFrame(eval_labels)
    eval_df.to_csv('visualizations\\eval_sample_per_epoch.csv', index=False)
    # plot train/val loss and acc
    plot_curves(train_losses, train_accuracies, val_losses, val_accuracies)