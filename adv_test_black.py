import argparse
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import genotypes
import torch.backends.cudnn as cudnn
from model import NetworkCIFAR
import torchattacks


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--test-batch-size', type=int, default=60, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--data_type', type=str, default='cifar10', help='which dataset to use')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
cudnn.benchmark = True


if args.data_type == 'cifar10':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR10(root='/home/fengyuqi/data', train=False, download=True, transform=transform_test)
elif args.data_type == 'cifar100':
    transform_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(transform_list)
    testset = torchvision.datasets.CIFAR100(root='/home/fengyuqi/data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


def cal_acc(model, X, y):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    return err


def eval_adv_acc(model_source, model_target, test_loader):
    model_source.eval()
    model_target.eval()
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        attack = torchattacks.PGD(model_source, eps=8/255, alpha=2/255, steps=20)
        adv_images = attack(data, target)
        X, y = Variable(adv_images, requires_grad=True), Variable(target)
        err_natural = cal_acc(model_target, X, y)
        natural_err_total += err_natural
        #print('batch err: ', natural_err_total)
    print('natural_err_total: ', natural_err_total)
    return natural_err_total


def main():
    if args.data_type == 'cifar100':
        CIFAR_CLASSES = 100
    elif args.data_type == 'cifar10':
        CIFAR_CLASSES = 10

    model_darts = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, False, genotypes.DARTS)
    model_pdarts = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, False, genotypes.PDARTS)
    model_racl = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, False, genotypes.RACL)
    model_advrush = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, False, genotypes.ADVRUSH)
    model_e2rnas = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, False, genotypes.E2RNAS)
    model_lrnas_10 = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, True, genotypes.search_cifar10)
    model_lrnas_100 = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, True, genotypes.search_cifar100)

    model_darts.load_state_dict(torch.load('/home/fengyuqi/black_box/100/DARTS.pt', map_location=torch.device('cuda:2')))
    model_darts.drop_path_prob = args.drop_path_prob
    model_darts.cuda()

    model_pdarts.load_state_dict(torch.load('/home/fengyuqi/black_box/100/PDARTS.pt', map_location=torch.device('cuda:2')))
    model_pdarts.drop_path_prob = args.drop_path_prob
    model_pdarts.cuda()

    model_racl.load_state_dict(torch.load('/home/fengyuqi/black_box/100/RACL.pt', map_location=torch.device('cuda:2')))
    model_racl.drop_path_prob = args.drop_path_prob
    model_racl.cuda()

    model_advrush.load_state_dict(torch.load('/home/fengyuqi/black_box/100/ADVRUSH.pt', map_location=torch.device('cuda:2')))
    model_advrush.drop_path_prob = args.drop_path_prob
    model_advrush.cuda()

    model_e2rnas.load_state_dict(torch.load('/home/fengyuqi/black_box/100/E2RNAS.pt', map_location=torch.device('cuda:2')))
    model_e2rnas.drop_path_prob = args.drop_path_prob
    model_e2rnas.cuda()

    model_lrnas_10.load_state_dict(torch.load('/home/fengyuqi/black_box/100/LRNAS_cifar10.pt', map_location=torch.device('cuda:2')))
    model_lrnas_10.drop_path_prob = args.drop_path_prob
    model_lrnas_10.cuda()

    model_lrnas_100.load_state_dict(torch.load('/home/fengyuqi/black_box/100/LRNAS_cifar100.pt', map_location=torch.device('cuda:2')))
    model_lrnas_100.drop_path_prob = args.drop_path_prob
    model_lrnas_100.cuda()

    print('darts: ')
    model_source = model_darts
    '''
    #eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('pdarts: ')
    model_source = model_pdarts
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    #eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('racl: ')
    model_source = model_racl
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    #eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('advrush: ')
    model_source = model_advrush
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    #eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('e2rnas: ')
    model_source = model_e2rnas
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    #eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('model_lrnas_10: ')
    model_source = model_lrnas_10
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    #eval_adv_acc(model_source, model_lrnas_10, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_100, test_loader)

    print('model_lrnas_100: ')
    model_source = model_lrnas_100
    '''
    eval_adv_acc(model_source, model_darts, test_loader)
    eval_adv_acc(model_source, model_pdarts, test_loader)
    eval_adv_acc(model_source, model_racl, test_loader)
    eval_adv_acc(model_source, model_advrush, test_loader)
    eval_adv_acc(model_source, model_e2rnas, test_loader)
    '''
    eval_adv_acc(model_source, model_lrnas_10, test_loader)
    #eval_adv_acc(model_source, model_lrnas_100, test_loader)


if __name__ == '__main__':
    main()