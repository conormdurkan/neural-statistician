import torch

from torchvision.utils import save_image


def save_test_grid(inputs, samples, save_path, n=10):
    inputs = inputs.cpu().data.view(-1, 5, 3, 64, 64)[:n]
    samples = samples.cpu().data.view(-1, 5, 3, 64, 64)[:10]
    images = torch.cat((inputs, samples), dim=1).view(-1, 3, 64, 64)
    save_image(images, save_path, nrow=10)
