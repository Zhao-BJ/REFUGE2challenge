import random
import torchvision
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, Sampler
from torchvision import transforms, datasets
from collections import defaultdict


class DatasetWrapper(Dataset):
    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys()) + 1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k * self.batch_size
                batch_indices = indices[offset: offset + self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)), self.batch_size)
            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))
            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        else:
            return self.num_iterations


def load_dataset(data_dir, sample='default', **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = DatasetWrapper(datasets.ImageFolder(root=data_dir, transform=transform))

    if sample == 'default':
        get_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
    elif sample == 'pair':
        get_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
    else:
        raise Exception('Unknown sampling')

    dataloader = DataLoader(dataset, batch_sampler=get_sampler(dataset))
    return dataloader
