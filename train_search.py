import time
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import ranking

from torch.autograd import Variable
from model_search import Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--samples', type=int, default=10, help='number of samples for estimation')
parser.add_argument('--data', type=str, default='/home/yuqi/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
args = parser.parse_args()


CIFAR_CLASSES = 10


def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device: %d' % args.gpu)
    print('args: %s' % args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()

    ops = []
    for cell_type in ['normal', 'reduce']:
        for edge in range(model.num_edges):
            ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in range(0, model.num_ops)])
    ops = np.concatenate(ops)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    #train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size // 2,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    prev_values = [1e-3 * torch.randn(model.num_edges, model.num_ops).cuda(),
                   1e-3 * torch.randn(model.num_edges, model.num_ops).cuda()]

    time1 = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()
        print('epoch: %d, lr: %e' % (epoch, lr[0]))

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        print('train acc: %f' % train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        print('valid acc: %f' % valid_acc)

        if epoch >= 15:
            normal_values, reduce_values = ranking.compute_value(valid_queue, model, ops, args.samples)
            print('normal estimation values: ', normal_values)
            print('reduce estimation values: ', reduce_values)

            prev_values = ranking.update_alpha([normal_values, reduce_values], prev_values)
            model.arch_parameters[0] += prev_values[0]
            model.arch_parameters[1] += prev_values[1]
            print('normal cell: ', model.arch_parameters[0])
            print('reduction cell: ', model.arch_parameters[1])

            cur_genotype = ranking.ranking(model.arch_parameters[0], model.arch_parameters[1], threshold=2.0,
                                       classes=CIFAR_CLASSES)
            print('genotype for current epoch: ', cur_genotype)

        #utils.save(model, '/home/fengyuqi/search.pt')

        scheduler.step()
    time2 = time.time()
    print('total cost: ', time2 - time1)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, requires_grad=False).cuda(non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
