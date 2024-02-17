#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader
from detector.model import get_effnet_detector, load_model_weights, save_model_weights
import click


@click.command()
@click.option('--model-name', default='effnet1', help='Model to work with')
@click.option('--difficulty', default=0.6, help='JIT difficulty')
@click.option('--length', default=12, help="number of iterations")
def eval(model_name, difficulty, length):
    model = get_effnet_detector()
    model_loc = f'weights/{model_name}.pth'
    model = load_model_weights(model, model_loc)

    from detector.loader import JITDataset
    dataset = JITDataset(length=length, difficulty=difficulty)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = torch.nn.MSELoss()

    # Assuming CUDA is available, use a GPU; otherwise, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Evaluate the average loss
    total_loss = 0.0
    with torch.no_grad():  # No need to track gradients for evaluation
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / length
    print(f'Average loss for a dozen inferences: {average_loss}')

if __name__ == "__main__":
    eval()