import numpy as np

from torch.utils import data


class SyntheticSetsDataset(data.Dataset):
    def __init__(self, n_datasets, sample_size, n_features, distributions):
        self.n_datasets = n_datasets
        self.sample_size = sample_size
        self.n_features = n_features
        self.distributions = distributions

        self.data = self.create_sets()
        self.datasets = self.data['datasets']

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return self.n_datasets

    def generate_distribution(self, distribution):
        m = np.random.uniform(-1, 1)
        v = np.random.uniform(0.5, 2)

        if distribution == 'gaussian':
            samples = np.random.normal(m, v, (self.sample_size, self.n_features))
            return samples, m, v

        elif distribution == 'mixture of gaussians':
            mix_1 = np.random.normal(-(1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
                                                               self.n_features))
            mix_2 = np.random.normal((1 + np.abs(m)), v / 2, (int(self.sample_size / 2),
                                                              self.n_features))
            return np.vstack((mix_1, mix_2)), 1 + np.abs(m), v / 2

        elif distribution == 'exponential':
            samples = np.random.exponential(1, (self.sample_size, self.n_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'reverse exponential':
            samples = - np.random.exponential(1, (self.sample_size, self.n_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'laplacian':
            samples = np.random.laplace(m, v, (self.sample_size, self.n_features))
            return samples, m, v

        elif distribution == 'uniform':
            samples = np.random.uniform(-1, 1, (self.sample_size, self.n_features))
            return self.augment_distribution(samples, m, v), m, v

        elif distribution == 'negative binomial':
            samples = np.random.negative_binomial(50, 0.5, (self.sample_size,
                                                            self.n_features))
            samples = np.asarray(samples, dtype=np.float64)
            return self.augment_distribution(samples, m, v), m, v

        else:
            print("Unrecognised choice of distribution.")
            return None

    @staticmethod
    def augment_distribution(samples, m, v):
        aug_samples = samples.copy()
        aug_samples -= np.mean(samples)
        aug_samples /= np.std(samples)
        aug_samples *= v ** 0.5
        aug_samples += m
        return aug_samples

    def create_sets(self):
        sets = np.zeros((self.n_datasets, self.sample_size, self.n_features),
                        dtype=np.float32)
        labels = []
        means = []
        variances = []

        for i in range(self.n_datasets):
            distribution = np.random.choice(self.distributions)

            x, m, v = self.generate_distribution(distribution)

            sets[i, :, :] = x
            labels.append(distribution)
            means.append(m)
            variances.append(v)

        return {
            "datasets": sets,
            "labels": np.array(labels),
            "means": np.array(means),
            "variances": np.array(variances),
            "distributions": np.array(self.distributions)
        }
