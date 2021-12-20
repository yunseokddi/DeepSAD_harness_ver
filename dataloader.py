import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_harness(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((64, 64))
    ])

    train = ImageFolder(root=args.train_dir, transform=transform)
    test = ImageFolder(root=args.test_dir, transform=transform)

    for data, target in test:
        print(target)

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
