import pandas as pd
import torch
import torch.utils.data
import torchvision


class Imbalanced_Dataset_Sampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) 

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        return dataset.get_multi_labels()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    def __call__(self):
        return self
    
from torch.utils.data import WeightedRandomSampler    
class Weighted_Random_Sampler():
    def __init__(
        self,
        dataset,
        labels
    ):
        self.class_counts = pd.DataFrame(labels).value_counts().to_list()
        self.num_samples = len(dataset)
        self.labels = labels
        self.class_weights = [self.num_samples / self.class_counts[i] for i in range(len(self.class_counts))]
        self.weights = [self.class_weights[labels[i]] for i in range(int(self.num_samples))]
    
    def __call__(self):
        return WeightedRandomSampler(torch.DoubleTensor(self.weights), int(self.num_samples)) 
    