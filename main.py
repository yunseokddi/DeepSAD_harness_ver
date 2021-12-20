import torch

from dataloader import get_harness
from options import Options
from torch.utils.tensorboard import SummaryWriter
from train import TrainerDeepSVDD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('./runs/experiment1')


def train():
    args = Options().parse()
    train_dataloader, test_dataloader = get_harness(args)
    deep_SVDD = TrainerDeepSVDD(args=args, data_loader=train_dataloader, device=device, R=0.0, nu=0.1, writer=writer)

    if args.pretrain:
        print("Start AUTOENCODER train!")
        deep_SVDD.pretrain()

    print("Start Deep SVDD train!")
    net, c = deep_SVDD.train()

    test_auroc = deep_SVDD.test(net=net, test_loader=test_dataloader)
    print("Test AUROC: {:.2f}".format(test_auroc * 100))
    writer.flush()
    writer.close()


if __name__ == "__main__":
    train()
