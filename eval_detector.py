#!/usr/bin/env python
import click
import torch
from torch.utils.data import DataLoader

from detector.model import get_effnet_detector, load_model_weights, get_vt_detector
from train_detector import get_transform_for_model


@click.command()
@click.option('--model-name', default='effnet1', help='Model to work with')
@click.option('--difficulty', default=0.6, help='JIT difficulty')
@click.option('--length', default=12, help="number of iterations")
def eval_model(model_name, difficulty, length):
    if 'effnet' in model_name:
        model = get_effnet_detector()
    elif 'vt' in model_name:
        model = get_vt_detector()

    model_loc = f'weights/{model_name}.pth'
    model = load_model_weights(model, model_loc)

    from detector.loader import JITDataset
    transform = get_transform_for_model(model_name)
    dataset = JITDataset(transform, length=length, difficulty=difficulty)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = torch.nn.MSELoss()

    # Assuming CUDA is available, use a GPU; otherwise, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Evaluate the average loss
    total_loss = 0.0
    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / length
    print(f'Average loss for a dozen inferences: {average_loss}')


if __name__ == "__main__":
    eval_model()
