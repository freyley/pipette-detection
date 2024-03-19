#!/usr/bin/env python

import os
from PIL import Image
from torchvision import transforms
import click

@click.command()
@click.option('--input_folder', '-i', default='./train/', help='Path to the input folder containing images')
@click.option('--output_folder', '-o', default='./train_224/', help='Path to the output folder for transformed images')
def main(input_folder, output_folder):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.png', '.bmp')):
            # Open the image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Apply the transformation
            transformed_image = transform(image)

            # Convert the normalized tensor back to a PIL Image
            transformed_image = transforms.ToPILImage()(transformed_image)

            # Save the transformed image to the output folder
            output_path = os.path.join(output_folder, filename)
            transformed_image.save(output_path)

    print('Transformation complete!')

if __name__ == "__main__":
    main()