import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import genotypes
from torch.cuda.amp import GradScaler, autocast
from torchattacks.attack import Attack

from torch.autograd import Variable
from model import NetworkImageNet as Network


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--gpus', type=list, default=[2, 3], help='gpu device id')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--data', type=str, default='/data/xiaotian/dataset/Imagenet/', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


CLASSES = 1000
scaler = GradScaler()
main_device = torch.device('cuda:' + str(args.gpus[0]))


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('args: %s' % args)

    genotype = genotypes.search_cifar10
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model.to(main_device)
    model = nn.DataParallel(model, device_ids=args.gpus)

    print("param size: ", utils.count_parameters_in_MB(model), 'MB')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(main_device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_imagenet1K()
    train_data = dset.ImageFolder(root=args.data + 'train', transform=train_transform)
    valid_data = dset.ImageFolder(root=args.data + 'val', transform=valid_transform)
    train_queue = DataLoaderX(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    valid_queue = DataLoaderX(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    attack = FastFGSMTrain(model=model, eps=4/255)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('epoch: %d' % epoch)

        train_acc, loss = train(train_queue, model, criterion, optimizer, attack)
        print('train_acc: ', train_acc)
        print('training loss: ', loss)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        print('valid_acc: ', valid_acc)

        utils.save(model, '/home/yuqi/LW_R3/train_imagenet/model_adv_' + str(epoch) + '.pt')


def train(train_loader, model, criterion, optimizer, attack):
    # Initialize the meters
    losses = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # switch to train mode
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(main_device, non_blocking=True)
        targets = targets.to(main_device, non_blocking=True)
        inputs = attack(inputs, targets)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, requires_grad=False).to(main_device, non_blocking=True)
            target = Variable(target, requires_grad=False).to(main_device, non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def adjust_learning_rate(optimizer, epoch):
    lr = args.learning_rate
    if epoch >= 20:
        lr = args.learning_rate * 0.1
    if epoch >= 25:
        lr = args.learning_rate * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class FastFGSMTrain(Attack):

    def __init__(self, model, eps):
        super().__init__("FastFGSMTrain", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.scaler = GradScaler()
        self.alpha = self.return_alpha()

    def return_alpha(self):
        self.child = []
        for name, child in self.model.module.named_children():
            self.child.append(child)

        if self.child[-1].out_features > 10:
            return 2.5
        else:
            return 1.25

    def forward(self, images, labels):
        images = images.clone().detach().to(main_device, non_blocking=True)
        labels = labels.clone().detach().to(main_device, non_blocking=True)

        adv_images = images.clone().detach()
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)

        loss = torch.nn.CrossEntropyLoss()

        adv_images.requires_grad = True

        # Accelarating forward propagation
        with autocast():
            outputs = self.model(adv_images)
            # Calculate loss
            cost = loss(outputs, labels)

        # Accelerating Gradient
        scaled_loss = self.scaler.scale(cost)
        # Update adversarial images
        grad = torch.autograd.grad(scaled_loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images_ = adv_images.detach() + self.alpha * self.eps*grad.sign()
        delta = torch.clamp(adv_images_ - images, min=-self.eps, max=self.eps)
        return torch.clamp(images + delta, min=0, max=1).detach()


if __name__ == '__main__':
    main()