import numpy as np

from torch.utils import data


class SyntheticSetsDataset(data.Dataset):
    def __init__(self, n_datasets, sample_size, n_features, distributions):
        self._n_datasets = n_datasets
        self._sample_size = sample_size
        self._n_features = n_features
        self._distributions = distributions

        self.data = self._create_sets()
        self.datasets = self.data['datasets']

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return self._n_datasets

    def _generate_distribution(self, distribution):
        m = np.random.uniform(-1, 1)
        v = np.random.uniform(0.5, 2)

        if distribution == 'gaussian':
            samples = np.random.normal(m, v, (self._sample_size, self._n_features))
            return samples, m, v

        elif distribution == 'mixture of gaussians':
            mix_1 = np.random.normal(-(1 + np.abs(m)), v / 2, (int(self._sample_size / 2),
                                                               self._n_features))
            mix_2 = np.random.normal((1 + np.abs(m)), v / 2, (int(self._sample_size / 2),
                                                              self._n_features))
            return np.vstack((mix_1, mix_2)), 1 + np.abs(m), v / 2

        elif distribution == 'exponential':
            samples = np.random.exponential(1, (self._sample_size, self._n_features))
            return self._augment_distribution(samples, m, v), m, v

        elif distribution == 'reverse exponential':
            samples = - np.random.exponential(1, (self._sample_size, self._n_features))
            return self._augment_distribution(samples, m, v), m, v

        elif distribution == 'laplacian':
            samples = np.random.laplace(m, v, (self._sample_size, self._n_features))
            return samples, m, v

        elif distribution == 'uniform':
            samples = np.random.uniform(-1, 1, (self._sample_size, self._n_features))
            return self._augment_distribution(samples, m, v), m, v

        elif distribution == 'negative binomial':
            samples = np.random.negative_binomial(50, 0.5, (self._sample_size,
                                                            self._n_features))
            samples = np.asarray(samples, dtype=np.float64)
            return self._augment_distribution(samples, m, v), m, v

        else:
            print("Unrecognised choice of distribution.")
            return None

    @staticmethod
    def _augment_distribution(samples, m, v):
        aug_samples = samples.copy()
        aug_samples -= np.mean(samples)
        aug_samples /= np.std(samples)
        aug_samples *= v ** 0.5
        aug_samples += m
        return aug_samples

    def _create_sets(self):
        sets = np.zeros((self._n_datasets, self._sample_size, self._n_features),
                        dtype=np.float32)
        labels = []
        means = []
        variances = []

        for i in range(self._n_datasets):
            distribution = np.random.choice(self._distributions)

            x, m, v = self._generate_distribution(distribution)

            sets[i, :, :] = x
            labels.append(distribution)
            means.append(m)
            variances.append(v)

        return {
            "datasets": sets,
            "labels": np.array(labels),
            "means": np.array(means),
            "variances": np.array(variances),
            "distributions": np.array(self._distributions)
        }