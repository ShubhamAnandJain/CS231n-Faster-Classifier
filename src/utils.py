import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.models as models
from torch.utils.data.sampler import Sampler

def get_accuracy(loader, model, device, dtype): 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return  100*acc, loss.item()

dataset_dict = {
  'CIFAR10' : torchvision.datasets.CIFAR10,
  'CIFAR100' : torchvision.datasets.CIFAR100,
  'ImageNet' : torchvision.datasets.ImageNet,
  'FashionMNIST' : torchvision.datasets.FashionMNIST,
  'MNIST' : torchvision.datasets.MNIST,
  'Places365' : torchvision.datasets.Places365,
}

def get_normalized_transform(dataset_name, train_ratio):

  transformToTensor = transforms.Compose(
    [transforms.ToTensor()])

  trainset = dataset_dict[dataset_name](root='./data', train=True,
                                        download=True, transform=transformToTensor)

  dataset_size = len(trainset)
  num_train = int(train_ratio * dataset_size)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_train, num_workers=1)
  data = next(iter(trainloader))
  mean, std_dev = data[0].mean(), data[0].std()

  normalized_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std_dev)])

  print(f'{dataset_name} Train dataset raw mean: {mean}, raw std dev: {std_dev}')

  return normalized_transform


class ImportanceSamplingDatasetWrapper():
  def __init__(self, dset):
    self.dset = dset

  def __getitem__(self, idx):
    x, y = self.dset.__getitem__(idx)
    return x, y, idx


def load_dataset(dataset_name, train_ratio, batch_size, train_sampler=None):

  transform = get_normalized_transform(dataset_name, train_ratio)

  trainset = dataset_dict[dataset_name](root='./data', train=True,
                                          download=True, transform=transform)

  dataset_size = len(trainset)
  num_train = int(train_ratio * dataset_size)

  if train_sampler is None:
    train_sampler = sampler.SubsetRandomSampler(range(num_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                            sampler=train_sampler)
  else:
    trainset = ImportanceSamplingDatasetWrapper(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, num_workers=2,
                                            batch_sampler=train_sampler)
  
  valset = dataset_dict[dataset_name](root='./data', train=True,
                                        download=True, transform=transform)
  valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=2,
                                           sampler=sampler.SubsetRandomSampler(range(num_train, dataset_size)))

  testset = dataset_dict[dataset_name](root='./data', train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2)

  print(f'INFO: Size of dataset: Training {num_train}, Validation {dataset_size - num_train}, Test {len(testset)}')

  num_channels = trainset[0][0].shape[0]

  return trainloader, valloader, testloader, num_train, num_channels


class ImportanceSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, num_train, weight, batch_size):
      
        self.weight = weight
        self.batch_size = batch_size
        self.num_batches = num_train // self.batch_size
        self.num_train = num_train

    def __iter__(self):
        while self.num_batches > 0:
          sampled = torch.split(torch.multinomial(self.weight, self.num_train, replacement=True), self.batch_size)
          return iter(sampled)

    def __len__(self):
        return self.num_train