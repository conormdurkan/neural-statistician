import torch

from torchvision.utils import save_image


def save_test_grid(inputs, samples, save_path, n=10):
    inputs = 1 - inputs.cpu().data.view(-1, 5, 1, 28, 28)[:n]
    reconstructions = samples.cpu().data.view(-1, 5, 1, 28, 28)[:n]
    images = torch.cat((inputs, reconstructions), dim=1).view(-1, 1, 28, 28)
    save_image(images, save_path, nrow=n)
