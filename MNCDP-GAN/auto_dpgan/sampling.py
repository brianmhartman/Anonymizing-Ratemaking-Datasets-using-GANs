from torch.utils.data import DataLoader, Sampler
import numpy as np


class IIDBatchSampler(Sampler):
    def __init__(self, dataset, minibatch_size, iterations):
        super(Sampler, self).__init__()
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            selections = np.random.binomial(self.length,
                                            self.minibatch_size / self.length)
            indices = np.random.choice(self.length, selections)
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


class EquallySizedAndIndependentBatchSampler(Sampler):
    def __init__(self, dataset, minibatch_size, iterations):
        super(Sampler, self).__init__()
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size)

    def __len__(self):
        return self.iterations


def get_data_loaders(minibatch_size, microbatch_size, iterations,
                     nonprivate=False, n_cpus=0):
    if not nonprivate:
        def minibatch_loader(dataset):
            return DataLoader(
                dataset,
                batch_sampler=IIDBatchSampler(dataset,
                                              minibatch_size,
                                              iterations),
                num_workers=n_cpus,
            )

        def microbatch_loader(minibatch):
            return DataLoader(
                minibatch,
                batch_size=microbatch_size,
                # drop_last=True,  # Creates NaN when microBS = miniBS
                num_workers=n_cpus,
            )
    else:
        def minibatch_loader(dataset):
            return DataLoader(
                dataset,
                batch_sampler=(EquallySizedAndIndependentBatchSampler
                               (dataset, minibatch_size, iterations)),
                num_workers=n_cpus,
            )

        def microbatch_loader(minibatch):
            return DataLoader(
                minibatch,
                batch_size=minibatch_size,
                num_workers=n_cpus,
            )

    return minibatch_loader, microbatch_loader
