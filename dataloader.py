import torch
import torch.utils.data as data
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class HarnessLoader(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('/')[-2]

        if label == "glue":
            label = 0
        elif label == "defect":
            label = 1

        return img_transformed, label

    def __len__(self):
        return len(self.file_list)


def get_file_list(img_path):
    classes = os.listdir(img_path)
    glues = []
    defects = []

    if img_path.split('/')[-2] == 'train':
        full_path = img_path + classes[0]
        for file_name in os.listdir(full_path):
            glues.append(os.path.join(full_path, file_name))
        return glues

    # classes = [defect, glue]
    else:
        full_path = img_path + classes[1]
        for file_name in os.listdir(full_path):
            glues.append(os.path.join(full_path, file_name))

        full_path = img_path + classes[0]
        for file_name in os.listdir(full_path):
            glues.append(os.path.join(full_path, file_name))

        return glues + defects


def get_harness(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64, 64))
    ])

    train_files = get_file_list(args.train_dir)
    test_files = get_file_list(args.test_dir)

    train = HarnessLoader(train_files, transform)
    test = HarnessLoader(test_files, transform)

    print('Num of train dataset: {}'.format(len(train)))
    print('Num of test dataset: {}'.format(len(test)))

    dataloder_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloder_test = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return dataloder_train, dataloder_test


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x