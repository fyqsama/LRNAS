import argparse
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import genotypes
import torch.backends.cudnn as cudnn
from model import NetworkCIFAR, NetworkTinyImageNet
import torchattacks


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--test-batch-size', type=int, default=75, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
cudnn.benchmark = True


if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='/home/yuqi/data', train=False, download=True, transform=transform_test)
elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='/home/yuqi/data', train=False, download=True, transform=transform_test)
elif args.data_type == 'tinyimagenet':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.ImageFolder(root='/home/yuqi/data/tiny-imagenet-200/val', transform=transform_test)


test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


def cal_acc(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    return err


def eval_adv_acc(model, test_loader, type):
    model.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
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
    if args.data_type == 'cifar100':
        CIFAR_CLASSES = 100
    elif args.data_type == 'cifar10':
        CIFAR_CLASSES = 10

    genotype = genotypes.search_cifar10
    if args.data_type == 'cifar10' or args.data_type == 'cifar100':
        model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        model.drop_path_prob = args.drop_path_prob
    else:
        model = NetworkTinyImageNet(C=48, num_classes=200, layers=14, auxiliary=True, genotype=genotype)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    model.load_state_dict(torch.load('/home/yuqi/LW_R3/model.pt', map_location=torch.device('cuda:' + str(args.gpu))))
    model = model.cuda()
    eval_adv_acc(model, test_loader, 'FGSM')
    eval_adv_acc(model, test_loader, 'PGD20')
    eval_adv_acc(model, test_loader, 'CW')


if __name__ == '__main__':
    main()