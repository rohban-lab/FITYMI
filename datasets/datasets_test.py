import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from datasets.datasets_train import get_nomral_dataset
from datasets.utils import sparse_to_coarse, BaseDataset


def get_test_transforms():
    transform_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_test


def get_normal_dataset_test(dataset_name, label, data_path, download, batch_size):
    normal_train_ds = get_nomral_dataset(dataset_name, label, data_path, download, get_test_transforms())
    normal_train_loader = DataLoader(normal_train_ds, batch_size=batch_size, shuffle=False)
    return normal_train_loader


def get_test_loader_one_vs_all(dataset_name, label, data_path, download, batch_size):
    traget_transform_func = lambda t: int(t != label)
    if dataset_name == 'cifar10':
        test_ds = CIFAR10(data_path, train=False, download=download, transform=get_test_transforms(),
                          target_transform=traget_transform_func)
    elif dataset_name == 'cifar100':
        test_ds = CIFAR100(data_path, train=False, download=download, transform=get_test_transforms(),
                           target_transform=traget_transform_func)
        test_ds.targets = sparse_to_coarse(test_ds.targets)
    else:
        raise NotImplementedError()

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def get_test_loader_one_vs_one(normal_label, ano_label, data_path, download, batch_size):
    test_cifar10 = CIFAR10(data_path, train=False, download=download)
    normal_test_data = test_cifar10.data[np.array(test_cifar10.targets) == normal_label]
    normal_test_ds = BaseDataset(normal_test_data, [0] * len(normal_test_data), get_test_transforms())

    test_cifar100 = CIFAR100(data_path, train=False, download=download)
    ano_test_data = test_cifar100.data[np.array(test_cifar100.targets) == ano_label]
    ano_test_ds = BaseDataset(ano_test_data, [1] * len(ano_test_data), get_test_transforms())

    test_ds = ConcatDataset([normal_test_ds, ano_test_ds])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader
