import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from datasets.utils import BaseDataset, sparse_to_coarse


def get_train_transforms():
    transform_train = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_train


def get_nomral_dataset(dataset_name, label, data_path, download, normal_transform):
    if dataset_name == 'cifar10':
        normal_train_ds = CIFAR10(data_path, train=True, download=download)
    elif dataset_name == 'cifar100':
        normal_train_ds = CIFAR100(data_path, train=True, download=download)
        normal_train_ds.targets = sparse_to_coarse(normal_train_ds.targets)
    else:
        raise NotImplementedError()
    normal_data = normal_train_ds.data[np.array(normal_train_ds.targets) == label]
    normal_train_ds = BaseDataset(normal_data, [0] * len(normal_data), normal_transform)
    return normal_train_ds


def get_gen_dataset(label, gen_data_path, gen_data_len):
    gen_images = np.load(gen_data_path)
    gen_images = gen_images[gen_data_len * label: gen_data_len * (label + 1)]
    gen_ds = BaseDataset(gen_images, [1] * len(gen_images), get_train_transforms())
    return gen_ds


def get_full_train_loader(args):
    normal_train_ds = get_nomral_dataset(args.dataset, args.label, args.normal_data_path, args.download_dataset,
                                         get_train_transforms())
    gen_ds = get_gen_dataset(args.label, args.gen_data_path, len(normal_train_ds))
    train_ds = ConcatDataset([normal_train_ds, gen_ds])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    return train_loader
