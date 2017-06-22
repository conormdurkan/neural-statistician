import numpy as np
import pickle

from skimage.transform import rotate
from torch.utils import data


class OmniglotSetsDataset(data.Dataset):
    def __init__(self, sample_size=5, split='train', augment=False):
        self.sample_size = sample_size
        path = '/home/conor/Dropbox/msc/thesis/data/omniglot/train_val_test_split.pkl'
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

        # assert len(sets) == len(set_labels)
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