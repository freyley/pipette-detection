#!/usr/bin/env python

# 1. find better arg parse for python - click?
# 2. move the model into an efficientnet model file
# 3. training script which saves weights to a file
#       loads weights from file
#       best loss gets saved
#       change learning rate as loss changes
# 4.  show loss on a log scale
from detector.model import get_effnet_detector, load_model_weights, save_model_weights
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import click
import os

def __train(model, loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode

    running_loss = 0.0
    count = 0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1  # Increment count for each batch
    return running_loss / count

def get_optimizer(loss, model):
    if loss > 10000:
        lr = 0.1
    elif loss > 1000:
        lr = 0.01
    else:
        lr = 0.001
    optimizer = torch.optim.Adam([
        {'params': model.classifier[1].parameters(), 'lr': lr}, # Higher learning rate for the new output layer
        {'params': model.features[0][0].parameters(), 'lr': lr},
    ], lr=1e-4)  # Default learning rate, in case there are parameters not included in any group
    return optimizer

def display_losses(losses):
    plt.figure(figsize=(10, 6))  # Set the figure size
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')  # Use the axes method to set label
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    plt.show()



@click.command()
@click.option('--batch-size', default=10, help='Batch size')
@click.option('--training_dir', default='./train', help='Directory to pull training files from')
@click.option('--train-jit', is_flag=True, help='Use the JIT training')
@click.option('--difficulty', default=0.6, help='JIT difficulty')
@click.option('--epochs', default=25, help='Epochs')
@click.option('--model-name', default='effnet1', help='Model to work with')
def train(batch_size, training_dir, train_jit, difficulty, epochs, model_name):
    model = get_effnet_detector()
    model_loc = f'weights/{model_name}.pth'
    model = load_model_weights(model, model_loc)

    from detector.loader import JITDataset, FileDataset
    if train_jit:
        dataset = JITDataset(length=100, difficulty=difficulty)
    else:
        dataset = FileDataset(img_dir=training_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    # start with a low learning rate optimizer because we don't know how far we are
    # in training
    optimizer = get_optimizer(1, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    losses = []

    for epoch in tqdm(range(epochs)):
        loss = __train(model, loader, criterion, optimizer, device)
        losses.append(loss)
        # probably don't need to do this _every_ time...
        optimizer = get_optimizer(loss, model)

        # checkpoint frequently on big training runs
        if epochs > 500 and epoch % 100 == 0:
            save_model_weights(model, model_loc)

    save_model_weights(model, model_loc)

    display_losses(losses)

if __name__ == '__main__':
    train()
