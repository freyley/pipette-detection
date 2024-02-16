import csv

import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from make_training_data import make_training_data, PipetteTemplate

TRANSFORMS = transforms.Compose([
    transforms.Grayscale(),  # Converts to grayscale
    transforms.Resize((500, 500)),  # Ensures image is 500x500
    transforms.ToTensor(),  # Converts to tensor
])


class JITDataset(Dataset):
    def __init__(self, length=1000, difficulty=0):
        self.length = length
        self.difficulty=difficulty
        self.transform = TRANSFORMS

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image, pip_pos = make_training_data(
            shape=(500, 500),
            template=PipetteTemplate('yip_2019_template.npz'),
            difficulty=self.difficulty,
        )
        # TODO: am I just deconverting and then grayscaling later? wtf?
        image = Image.fromarray(image*255).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(pip_pos, dtype=torch.float)
        return image, target


def csv_to_dict(csv_filepath):
    result_dict = {}
    with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Ensure the row has at least 4 elements
            if len(row) >= 4:
                key = row[0]
                value = row[1:4]  # Get the next three elements
                result_dict[key] = value
    return result_dict


class FileDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = TRANSFORMS
        self.img_names = list(filter(lambda x: x.endswith('.jpg'), os.listdir(img_dir)))
        self.points_dict = csv_to_dict(os.path.join(self.img_dir, 'pos.csv'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)  # Load as PIL Image
        if self.transform:
            image = self.transform(image)

        target_coords = [float(coord) for coord in self.points_dict[img_name]]
        target = torch.tensor(target_coords, dtype=torch.float)

        return image, target
