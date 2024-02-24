#!/usr/bin/env python

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from detector.model import get_effnet_detector, get_vt_detector, load_model_weights, save_model_weights


# 1. find better arg parse for python - click?
# 2. move the model into an efficientnet model file
# 3. training script which saves weights to a file
#       loads weights from file
#       best loss gets saved
#       change learning rate as loss changes
# 4.  show loss on a log scale


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


def get_optimizer(loss, model, frozen=False, no_pretrained=False):
    if frozen:
        default_lr = 0
    else:
        default_lr = 1e-7
    if loss > 1000:
        lr = 1e-3
    else:
        lr = 1e-5
    if no_pretrained:
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(
            [
                {
                    'params': model.get_last_layer().parameters(),
                    'lr': lr,
                },  # Higher learning rate for the new output layer
            ],
            lr=default_lr,
        )


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


def get_transform_for_model(model_name: str):
    if 'effnet' in model_name:
        return transforms.Compose([
            # transforms.Grayscale(),  # Converts to grayscale # not doing that anymore
            transforms.Resize((500, 500)),  # Ensures image is 500x500
            transforms.ToTensor(),  # Converts to tensor
        ])
    elif 'vt' in model_name:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels, VTs don't support 500x500
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Unknown model! vt or effnet only so far!")


@click.command()
@click.option('--batch-size', default=10, help='Batch size')
@click.option('--training_dir', default='./train', help='Directory to pull training files from')
@click.option('--train-jit', is_flag=True, help='Use the JIT training')
@click.option('--difficulty', default=0.3, help='JIT difficulty')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model-name', help='Model to work with')
@click.option('--frozen', is_flag=True, help="Freeze early layers")
@click.option('--no-pretrained', is_flag=True, help="Don't use pretrained weights")
@click.option('--size', default=None, help="model size - m or l for effnet, b16, l16, l32 for vt")
def train(batch_size, training_dir, train_jit, difficulty, epochs, model_name, frozen, no_pretrained, size):
    if 'effnet' in model_name:
        model = get_effnet_detector(no_pretrained, size)
    elif 'vt' in model_name:
        model = get_vt_detector(no_pretrained, size)
    else:
        raise ValueError("Unknown model! vt or effnet only so far!")
    transform = get_transform_for_model(model_name)
    model_loc = f'weights/{model_name}.pth'
    model = load_model_weights(model, model_loc)

    from detector.loader import JITDataset, FileDataset
    if train_jit:
        dataset = JITDataset(transform, length=100, difficulty=difficulty)
    else:
        dataset = FileDataset(img_dir=training_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    # start with a low learning rate optimizer because we don't know how far we are
    # in training
    optimizer = get_optimizer(1, model, frozen=frozen, no_pretrained=no_pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    losses = []

    for epoch in tqdm(range(epochs)):
        loss = __train(model, loader, criterion, optimizer, device)
        losses.append(loss)
        # probably don't need to do this _every_ time...
        optimizer = get_optimizer(loss, model, frozen=frozen, no_pretrained=no_pretrained)

        # checkpoint frequently on big training runs
        if epochs > 500 and epoch % 100 == 0:
            save_model_weights(model, model_loc)

    save_model_weights(model, model_loc)

    display_losses(losses)


if __name__ == '__main__':
    train()
