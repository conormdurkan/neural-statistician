import gzip
import numpy as np
import os
import pickle
import torch

from skimage.transform import rotate
from torch.utils import data


def load_mnist(data_dir):

    def load_images(path):
        with gzip.open(path) as bytestream:
            # read meta information
            header_buffer = bytestream.read(16)
            header = np.frombuffer(header_buffer, dtype='>i4')
            magic, n, x, y = header
            # read data
            buffer = bytestream.read(x * y * n)
            data = np.frombuffer(buffer, dtype='>u1').astype(np.float32)
            data = data.reshape(n, x * y)
        return data

    def load_labels(path):
        with gzip.open(path) as bytestream:
            # read meta information
            header_buffer = bytestream.read(8)
            header = np.frombuffer(header_buffer, dtype='>i4')
            magic, n = header
            # read data
            buffer = bytestream.read(n)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)
        return data

    train_images_gz = 'train-images-idx3-ubyte.gz'
    train_labels_gz = 'train-labels-idx1-ubyte.gz'

    test_images_gz = 't10k-images-idx3-ubyte.gz'
    test_labels_gz = 't10k-labels-idx1-ubyte.gz'

    train_images = load_images(os.path.join(data_dir, train_images_gz))
    test_images = load_images(os.path.join(data_dir, test_images_gz))
    images = np.vstack((train_images, test_images)) / 255

    train_labels = load_labels(os.path.join(data_dir, train_labels_gz))
    test_labels = load_labels(os.path.join(data_dir, test_labels_gz))
    labels = np.hstack((train_labels, test_labels))

    n = len(labels)
    one_hot = np.zeros((n, 10))
    one_hot[range(n), labels] = 1
    labels = one_hot

    return images, labels


def load_mnist_test_batch(data_dir, batch_size):
    images, one_hot_labels = load_mnist(data_dir=data_dir)
    labels = np.argmax(one_hot_labels, axis=1)
    ixs = [np.random.choice(np.where(labels == i)[0], size=5, replace=False)
           for i in range(10)]
    batch = np.array([images[ix] for ix in ixs])
    return torch.from_numpy(batch).clone().repeat((batch_size // 10) + 1, 1, 1)[:batch_size]


class OmniglotSetsDataset(data.Dataset):
    def __init__(self, data_dir, sample_size=5, split='train', augment=False):
        self.sample_size = sample_size
        path = os.path.join(data_dir, 'train_val_test_split.pkl')
        with open(path, 'rb') as file:
            splits = pickle.load(file)
        if split == 'train':
            images, labels = splits[:2]
            sets, set_labels = self.make_sets(images, labels)
            if augment:
                sets = self.augment_sets(sets)
            else:
                sets = np.random.binomial(1, p=sets, size=sets.shape).astype(np.float32)
        elif split == 'valid':
            images, labels = splits[2:4]
            sets, set_labels = self.make_sets(images, labels)
        elif split == 'test':
            images, labels = splits[4:]
            sets, set_labels = self.make_sets(images, labels)
        else:
            "Unrecognized split, returning None."
            sets, set_labels = None, None
        sets = sets.reshape(-1, 5, 1, 28, 28)

        self.n = len(sets)
        self.data = {
            'inputs': sets,
            'targets': set_labels
        }

    def __getitem__(self, item):
        return self.data['inputs'][item]

    def __len__(self):
        return self.n

    def augment_sets(self, sets):
        augmented = np.copy(sets)
        augmented = augmented.reshape(-1, self.sample_size, 28, 28)
        n_sets = len(augmented)

        for s in range(n_sets):
            flip_horizontal = np.random.choice([0, 1])
            flip_vertical = np.random.choice([0, 1])
            if flip_horizontal:
                augmented[s] = augmented[s, :, :, ::-1]
            if flip_vertical:
                augmented[s] = augmented[s, :, ::-1, :]

        for s in range(n_sets):
            angle = np.random.uniform(0, 360)
            for item in range(self.sample_size):
                augmented[s, item] = rotate(augmented[s, item], angle)
        augmented = np.concatenate([augmented.reshape(n_sets, self.sample_size, 28*28),
                                    sets])

        return augmented

    @staticmethod
    def one_hot(dense_labels, num_classes):
        num_labels = len(dense_labels)
        offset = np.arange(num_labels) * num_classes
        one_hot_labels = np.zeros((num_labels, num_classes))
        one_hot_labels.flat[offset + dense_labels.ravel()] = 1
        return one_hot_labels

    def make_sets(self, images, labels):
        num_classes = np.max(labels) + 1
        labels = self.one_hot(labels, num_classes)

        n = len(images)
        perm = np.random.permutation(n)
        images = images[perm]
        labels = labels[perm]

        image_sets = []
        set_labels = []

        for i in range(num_classes):
            ix = labels[:, i].astype(bool)
            num_instances_of_class = np.sum(ix)
            if num_instances_of_class < self.sample_size:
                pass
            else:
                remainder = num_instances_of_class % self.sample_size
                image_set = images[ix]
                if remainder > 0:
                    image_set = image_set[:-remainder]
                image_sets.append(image_set)
                k = len(image_set)
                set_labels.append(labels[ix][:int(k / self.sample_size)])

        x = np.concatenate(image_sets, axis=0).reshape(-1, self.sample_size, 28*28)
        y = np.concatenate(set_labels, axis=0)
        if np.max(x) > 1:
            x /= 255

        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]

        return x, y