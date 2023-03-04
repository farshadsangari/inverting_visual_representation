import re
import random
import pandas as pd
from .prepare_data import get_file_list
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


def data_transform():
    data_transforms = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return data_transforms


class myDataset(Dataset):
    def __init__(self, images_directory, transform):
        self.img_files = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        rgb_image = image.convert("RGB")
        return (self.transform(rgb_image), self.transform(rgb_image))


def train_val_loader(train_pathes, val_pathes, data_transform, batch_size):

    train_images_directory = list(train_pathes["path"].values)
    val_images_directory = list(val_pathes["path"].values)

    train_data = myDataset(train_images_directory, data_transform)
    val_data = myDataset(val_images_directory, data_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def sample_loader(train_pathes, val_pathes, random_categories, data_transform):

    path_for_plot_list = []
    randm_class_for_plot = random.sample(random_categories, 5)
    for random_class in randm_class_for_plot:
        trn = train_pathes[train_pathes.category == random_class].sample(1)
        path_for_plot_list.append(trn)

    for random_class in randm_class_for_plot:
        tst = val_pathes[val_pathes.category == random_class].sample(1)
        path_for_plot_list.append(tst)

    path_for_plot = pd.concat(path_for_plot_list).reset_index(drop=True)
    path_for_plot_directory = list(path_for_plot["path"].values)
    plot_sample_data = myDataset(path_for_plot_directory, data_transform)
    sample_plot_loader = DataLoader(
        dataset=plot_sample_data, batch_size=10, shuffle=False
    )

    return sample_plot_loader
