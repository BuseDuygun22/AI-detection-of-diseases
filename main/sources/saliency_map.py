import torch
import matplotlib.pyplot as plt
import numpy as np
from dc1.net import Net
from dc1.image_dataset import ImageDataset
from pathlib import Path

def generate_saliency_map(model_path: str, image_path: str, label_path: str):
    """
    Generates a saliency map for a given image from the dataset.
    :param model_path: Path to the trained model weights.
    :param image_path: Path to the dataset's image file (.npy format).
    :param label_path: Path to the dataset's label file (.npy format).
    """
    # Load model
    model = Net(n_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load image data
    images = np.load(image_path)
    labels = np.load(label_path)

    # Convert `images` from a NumPy array to a PyTorch tensor
    images_tensor = torch.tensor(images, dtype=torch.float32)

    for index in range(5):  # Iterate over a few images
        # Fix input shape to be [batch_size, channels, height, width] = [1, 1, 128, 128]
        image = torch.tensor(images[index], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128]
        label = labels[index]
        image.requires_grad = True  # Enable gradients

        # Forward pass for classification
        output = model(image)
        score, pred_label = output.max(1)
        model.zero_grad()
        score.backward()

        # Generate saliency map
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().detach().cpu().numpy()

        # Plot original image and saliency map
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image.squeeze().detach().cpu().numpy(), cmap='gray')
        axes[0].set_title(f'Original Image (Label: {label})')

        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title(f'Saliency Map (Pred: {pred_label.item()})')

        plt.show()


