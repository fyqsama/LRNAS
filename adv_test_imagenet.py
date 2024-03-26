import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
from prefetch_generator import BackgroundGenerator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import genotypes
import torch.backends.cudnn as cudnn
from model import NetworkImageNet
import torchattacks


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--gpus', type=list, default=[2, 3], help='gpu device id')
parser.add_argument('--batch_size', type=int, default=400, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--data', type=str, default='/data/xiaotian/dataset/Imagenet/', help='location of the data corpus')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
args = parser.parse_args()


cudnn.benchmark = True
CLASSES = 1000
main_device = torch.device('cuda:' + str(args.gpus[0]))


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def data_transforms_imagenet1K():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def cal_acc(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    return err


def eval_adv_acc(model, test_loader, type):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(main_device), target.to(main_device)
        if type == 'FGSM':
            attack = torchattacks.FGSM(model, eps=8/255)
        if type == 'PGD20':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        if type == 'PGD100':
            attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=100)
        if type == 'CW':
            attack = torchattacks.CW(model, c=0.5, steps=100)
        if type == 'APGD':
            attack = torchattacks.APGD(model, eps=8/255, steps=20)
        if type == 'AA':
            attack = torchattacks.AutoAttack(model, eps=8/255)
        adv_images = attack(data, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)
        err_natural = cal_acc(model, X, y)
        natural_err_total += err_natural
        print('batch err: ', natural_err_total)
    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def main():
    _, valid_transform = data_transforms_imagenet1K()
    valid_data = dset.ImageFolder(root=args.data + 'val', transform=valid_transform)
    test_loader = DataLoaderX(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    genotype = genotypes.search_cifar10
    model = NetworkImageNet(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model.to(main_device)
    model = nn.DataParallel(model, device_ids=args.gpus)

    model.load_state_dict(torch.load('/home/yuqi/LW_R3/train_imagenet/model_adv.pt', map_location=main_device))
    eval_adv_acc(model, test_loader, 'FGSM')
    eval_adv_acc(model, test_loader, 'PGD20')


if __name__ == '__main__':
    main()