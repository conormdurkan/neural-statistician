import argparse
import gzip
import numpy as np
import os
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, type=str, default=None)
args = parser.parse_args()
assert (args.data_dir is not None) and (os.path.isdir(args.data_dir))


# CREATE SPATIAL MNIST
def load_images(path):
    with gzip.open(path) as bytestream:
        # read meta information
        header_buffer = bytestream.read(16)
        header = np.frombuffer(header_buffer, dtype=">i4")
        magic, n, x, y = header
        # read data
        buffer = bytestream.read(x * y * n)
        data = np.frombuffer(buffer, dtype=">u1").astype(np.float32)
        data = data.reshape(n, x * y)
    return data


def load_labels(path):
    with gzip.open(path) as bytestream:
        # read meta information
        header_buffer = bytestream.read(8)
        header = np.frombuffer(header_buffer, dtype=">i4")
        magic, n = header
        # read data
        buffer = bytestream.read(n)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)
    return data


def load_data():
    train_images_gz = "train-images-idx3-ubyte.gz"
    train_labels_gz = "train-labels-idx1-ubyte.gz"

    test_images_gz = "t10k-images-idx3-ubyte.gz"
    test_labels_gz = "t10k-labels-idx1-ubyte.gz"

    data_dir = args.data_dir

    train_images = load_images(os.path.join(data_dir, train_images_gz))
    test_images = load_images(os.path.join(data_dir, test_images_gz))
    images = np.vstack((train_images, test_images))

    train_labels = load_labels(os.path.join(data_dir, train_labels_gz))
    test_labels = load_labels(os.path.join(data_dir, test_labels_gz))
    labels = np.hstack((train_labels, test_labels))

    n = len(labels)
    one_hot = np.zeros((n, 10))
    one_hot[range(n), labels] = 1
    labels = one_hot

    return images, labels


def create_spatial(images, labels, sample_size=50, plot=False):
    spatial = np.zeros([70000, sample_size, 2])
    grid = np.array([[i, j] for j in range(27, -1, -1) for i in range(28)])
    for i, image in enumerate(tqdm(images)):
        replace = True if (sum(image > 0) < sample_size) else False
        ix = np.random.choice(
            range(28 * 28),
            size=sample_size,
            p=image / sum(image),
            replace=replace,
        )
        spatial[i, :, :] = grid[ix] + np.random.uniform(0, 1, (sample_size, 2))

    # sanity check
    if plot:
        sample = spatial[:100]
        fig, axs = plt.subplots(10, 10, figsize=(8, 8))
        axs = axs.flatten()
        for i in range(100):
            axs[i].scatter(sample[i, :, 0], sample[i, :, 1], s=2)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlim([0, 27])
            axs[i].set_ylim([0, 27])
            axs[i].set_aspect("equal", adjustable="box")
        plt.show()

    spatial_path = os.path.join(args.data_dir, "spatial/spatial.pkl")
    with open(spatial_path, "wb") as file:
        pickle.dump(spatial, file)

    labels_path = os.path.join(args.data_dir, "spatial/labels.pkl")
    with open(labels_path, "wb") as file:
        pickle.dump(labels, file)


def main():
    images, labels = load_data()
    create_spatial(images, labels, sample_size=50, plot=False)


if __name__ == "__main__":
    main()
