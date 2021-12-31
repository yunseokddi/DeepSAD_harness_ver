import argparse


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train Deep SVDD model',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--num_epochs', '-e', type=int, default=100, help='Num of epochs to Deep SVDD train')
        self.parser.add_argument('--num_epochs_ae', '-ea', type=int, default=100, help='Num of epochs to AE model train')
        self.parser.add_argument('--lr', '-lr', type=float, default=1e-3, help='learning rate for model')
        self.parser.add_argument('--lr_ae', '-lr_ae', type=float, default=1e-3, help='learning rate for AE model')
        self.parser.add_argument('--weight_decay', '-wd', type=float, default=5e-7, help='weight decay for model')
        self.parser.add_argument('--weight_decay_ae', '-wd_ae', type=float, default=5e-3, help='weight decay for model')
        self.parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[50],
                                 help='learning rate milestones')
        self.parser.add_argument('--batch_size', '-bs', type=int, default=1024, help='batch size')
        self.parser.add_argument('--pretrain', '-pt', type=bool, default=True, help='Pretrain to AE model')
        self.parser.add_argument('--latent_dim', '-ld', type=int, default=32, help='latent dimension')
        self.parser.add_argument('--normal_class', '-cls', type=int, default=0, help='Set the normal class')
        self.parser.add_argument('--train_dir', '-train_dir', type=str,
                                 default='../data/harness_paper_dataset/code_dataset/4/train/', help='Train data path')
        self.parser.add_argument('--test_dir', '-test_dir', type=str,
                                 default='../data/harness_paper_dataset/code_dataset/4/test/', help='Test data path')

        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
