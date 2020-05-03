import torch
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
from torchvision.transforms import transforms

import glob
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler


class AttributeDiscoveryDataset(Dataset):
    def __init__(self, path_df, root_dir=None, transform=None):
        """
        Args:
            path_df (string): The DataFrame with all paths.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_df = path_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.path_df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.path_df.iloc[idx, 0]
        # image = io.imread(img_path)
        image = Image.open(img_path)
        if len(image.size) == 2:
            image = image.convert("RGB")
        label = self.path_df.iloc[idx, 1]
        desc_path = self.path_df.iloc[idx, 2]
        with open(desc_path, 'r') as file:
            desc = ''.join(file.read())
        sample = {'img': image, 'desc': desc, 'label': label}

        if self.transform:
            sample['img'] = self.transform(sample['img'])

        return sample


def get_dataloader(opt):
    if opt.dataset == "stl10":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
        return (
            train_loader,
            train_dataset,
            supervised_loader,
            supervised_dataset,
            test_loader,
            test_dataset,
        )
    elif opt.dataset == "attribute-discovery":
        # get list of images for each label
        img_paths = [None] * 4
        img_paths[0] = glob.glob(
            "/content/attribute-discovery-dataset/bags*/*/*.jpg")
        img_paths[1] = glob.glob(
            "/content/attribute-discovery-dataset/earrings*/*/*.jpg")
        img_paths[2] = glob.glob(
            "/content/attribute-discovery-dataset/ties*/*/*.jpg")
        img_paths[3] = glob.glob(
            "/content/attribute-discovery-dataset/womens*/*/*.jpg")
        unsupervised = []  # used for unsupervised training

        print("Before class balancing")
        print("Number of bags images: ", len(img_paths[0]))
        print("Number of earrings images: ", len(img_paths[1]))
        print("Number of ties images: ", len(img_paths[2]))
        print("Number of womens images: ", len(img_paths[3]))

        new_size = min(len(img_paths[0]), len(
            img_paths[1]), len(img_paths[2]), len(img_paths[3]))
        print("\nAfter class balancing, the size of each class is", new_size)

        def resize_dataset(dataset):
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            random_seed = 167
            np.random.seed(random_seed)
            np.random.shuffle(dataset)
            return dataset[:new_size], dataset[new_size:]

        for i in range(len(img_paths)):
            img_paths[i], set_aside = resize_dataset(img_paths[i])
            unsupervised.extend(set_aside)

        print("Before class balancing")
        print("Number of bags images: ", len(img_paths[0]))
        print("Number of earrings images: ", len(img_paths[1]))
        print("Number of ties images: ", len(img_paths[2]))
        print("Number of womens images: ", len(img_paths[3]))
        print("Number of unsupervised images:", len(unsupervised))

        img_label_list = []
        for i in range(4):
            for img_path in img_paths[i]:
                img_label_list.append([img_path, i])
        for img_path in unsupervised:
            img_label_list.append([img_path, 4])
        df = pd.DataFrame(img_label_list, columns=['img_path', 'label'])

        def img_path_to_desc_path(img_path):
            img_name = img_path.split('/')[-1]
            directory_path = '/'.join(img_path.split('/')[:-1])
            unique_sample_id = '_'.join(img_name.split('_')[1:])
            desc_path = directory_path + '/descr_' + unique_sample_id
            desc_path = desc_path[:-3] + 'txt'
            return desc_path

        if opt.load_descr:
            df['desc_path'] = df.apply(
                lambda row: img_path_to_desc_path(row['img_path']), axis=1)
        print(df.info())
        print(df.head())

        aug = {
            "ad": {
            "randcrop": 224,
            "flip": True,
            "grayscale": False,
            # values for train+unsupervised combined
            "mean": [0.4313, 0.4156, 0.3663],
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
        }

        dataset = AttributeDiscoveryDataset(
            df, transform=get_transforms(eval=False, aug=aug['ad'], dataset="attribute-discovery"))
        batch_size = 32
        test_split = .5
        shuffle_dataset = True
        random_seed = 42

        dataset_size = len(dataset) - len(unsupervised)
        print(dataset_size)
        indices = list(range(dataset_size))
        first_split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        test_indices, supervised_indices = indices[:
                                                   first_split], indices[first_split:]
        train_indices = [i + dataset_size for i in range(len(unsupervised))]
        print(len(test_indices), len(supervised_indices), len(train_indices))

        train_sampler = SubsetRandomSampler(train_indices)
        supervised_sampler = SubsetRandomSampler(supervised_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        supervised_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=supervised_sampler)
        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)
        return (
            train_loader,
            dataset,
            supervised_loader,
            None,
            test_loader,
            None,
        )
    else:
        raise Exception("Invalid option")


# def get_transforms(eval=False, aug=None):
#     trans = []

#     if aug["randcrop"] and not eval:
#         trans.append(transforms.RandomCrop(aug["randcrop"]))

#     if aug["randcrop"] and eval:
#         trans.append(transforms.CenterCrop(aug["randcrop"]))

#     if aug["flip"] and not eval:
#         trans.append(transforms.RandomHorizontalFlip())

#     trans.append(transforms.ToTensor())
#     # if aug["grayscale"]:
#     #     trans.append(transforms.Grayscale())
#     #     trans.append(transforms.ToTensor())
#     #     trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
#     # elif aug["mean"]:
#     #     trans.append(transforms.ToTensor())
#     #     trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
#     # else:
#     #     trans.append(transforms.ToTensor())

#     trans = transforms.Compose(trans)
#     return trans


def get_stl10_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
                "randcrop": 64,
                "flip": True,
                "grayscale": opt.grayscale,
                # values for train+unsupervised combined
                "mean": [0.4313, 0.4156, 0.3663],
                "std": [0.2683, 0.2610, 0.2687],
                "bw_mean": [0.4120],  # values for train+unsupervised combined
                "bw_std": [0.2570],
            }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=transform_train,
        download=opt.download_dataset,
    )  # set download to True to get the dataset

    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        if opt.training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(
                dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        elif opt.training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(
                dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        else:
            raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            base_folder,
            split=opt.training_dataset,
            transform=transform_valid,
            download=opt.download_dataset,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=valid_sampler,
            num_workers=16,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_transforms(eval=False, aug=None, dataset="stl10"):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if dataset == "stl10":
        if aug["grayscale"]:
            trans.append(transforms.Grayscale())
            trans.append(transforms.ToTensor())
            trans.append(transforms.Normalize(
                mean=aug["bw_mean"], std=aug["bw_std"]))
        elif aug["mean"]:
            trans.append(transforms.ToTensor())
            trans.append(transforms.Normalize(
                mean=aug["mean"], std=aug["std"]))
        else:
            trans.append(transforms.ToTensor())
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans
