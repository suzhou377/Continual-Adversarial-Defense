from __future__ import print_function
import os
import copy
import torch
import time
import argparse
import numpy as np
import logging
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from encoder_linear import encoder_linear
from function import EMAModel
from function import ProtoCrossAttn
from function import dual_mlp
from function import ConfidenceLabelLoss
from function import PrototypeSupConLoss
from pfn_bank_dual import cad_loss
from dataset import CIFAR10
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--train-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--ema-decay', default=0.9995, type=float, metavar='W')
parser.add_argument('--epsilon', default=8/255, help='radii of perturbation')
parser.add_argument('--pgd-step-size', default=2/255, help='perturb step size')
parser.add_argument('--fgsm-step-size', default=8/255, help='perturb step size')
parser.add_argument('--train-num-steps', default=10, help='train perturb number of steps')
parser.add_argument('--test-pgd-steps', default=10, help='test perturb number of steps')
parser.add_argument('--test-fgsm-steps', default=1, help='test perturb number of steps')
parser.add_argument('--theta1', type=int, default=6, metavar='S')
parser.add_argument('--theta2', type=int, default=6, metavar='S')
parser.add_argument('--tau', default=1, help='exponential move average')
parser.add_argument('--ema-epoch', default=41, type=int, help='start epoch of move_average training')
parser.add_argument('--proto-num', type=float, default=10, metavar='M')
parser.add_argument('--proto-dim', type=float, default=512, metavar='M')
parser.add_argument('--proto-momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./pfn_bank_dual', help='directory of model for saving checkpoint')
args = parser.parse_args()
# settings
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
makedir(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
set_seed(args)
def adjust_learning_rate(args, optimizer, epoch):
    '''decrease the learning rate'''
    lr = args.lr
    if epoch >= 40:
        lr = args.lr * 0.1
    if epoch >= 80:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class Logger(object):
    def __init__(self, path):
        self.logger = logging.getLogger()
        self.path = path
        self.set_file_logger()
    def set_file_logger(self):
        handler = logging.FileHandler(self.path, 'w+')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    def log(self, message):
        self.logger.info(message)
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10(root='./cifar10_data', train=True, download=False, transform=transform_train)
test_dataset = CIFAR10(root='./cifar10_data', train=False, download=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
'''define train function'''
def train(args, encoder, classifier, move_average, cross_fusion, confidence_mlp, task_mlp,
          train_loader, optimizer, prototype_contrastive, natural_proto_pool, robust_proto_pool, proto_momentum,
          rampup, unique_label, epoch, criterion_nll, criterion_kl):
    encoder.train()
    classifier.train()
    robust_accuracy_total, natural_accuracy_total = 0, 0
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss, robust_accuracy, natural_accuracy, previous_natural_pool, previous_robust_pool = cad_loss(encoder=encoder,
                                                           classifier=classifier,
                                                           move_average=move_average,
                                                           cross_fusion=cross_fusion,
                                                           confidence_mlp=confidence_mlp,
                                                           task_mlp=task_mlp,
                                                           x_natural=data,
                                                           y=target,
                                                           optimizer=optimizer,
                                                           prototype_contrastive=prototype_contrastive,
                                                           natural_proto_pool=natural_proto_pool,
                                                           robust_proto_pool=robust_proto_pool,
                                                           proto_momentum=proto_momentum,
                                                           unique_label=unique_label,
                                                           criterion_nll=criterion_nll,
                                                           criterion_kl=criterion_kl,
                                                           step_size=args.pgd_step_size,
                                                           epsilon=args.epsilon,
                                                           theta1=args.theta1,
                                                           theta2=args.theta2,
                                                           rampup=rampup,
                                                           epoch=epoch)
        natural_proto_pool, robust_proto_pool = previous_natural_pool, previous_robust_pool
        loss.backward()
        optimizer.step()
        if epoch < args.ema_epoch:
            move_average.update(epoch, ema_epoch=args.ema_epoch, decay=args.tau)
        else:
            pass
        robust_accuracy_total += robust_accuracy
        natural_accuracy_total += natural_accuracy
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_robust_accuracy = robust_accuracy_total / len(train_loader.dataset)
    train_natural_accuracy = natural_accuracy_total / len(train_loader.dataset)
    print('Training：train_natural_accuracy: {:.4f}, train_robust_accuracy: {:.4f}, lr:{:.4f}'
        .format(train_natural_accuracy, train_robust_accuracy, optimizer.param_groups[0]['lr']))
    return train_natural_accuracy, train_robust_accuracy, natural_proto_pool, robust_proto_pool
def pgd_whitebox(encoder,
                  classifier,
                  X,
                  y,
                  num_steps,
                  step_size,
                  epsilon=args.epsilon):
    out = classifier(encoder(X))
    natural_accuracy = (out.data.max(1)[1] == y.data).float().sum()
    x_adv = X.detach() + torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).cuda()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(classifier(encoder(x_adv)), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    robust_accuracy = (classifier(encoder(x_adv)).data.max(1)[1] == y.data).float().sum()
    return natural_accuracy, robust_accuracy
def eval_pgd_whitebox(encoder, classifier, test_loader, num_steps, step_size):
    encoder.eval()
    classifier.eval()
    natural_accuracy_total, robust_accuracy_total = 0, 0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        natural_accuracy, robust_accuracy = pgd_whitebox(encoder, classifier, X, y, num_steps, step_size)
        natural_accuracy_total += natural_accuracy
        robust_accuracy_total += robust_accuracy
    test_natural_accuracy = natural_accuracy_total / len(test_loader.dataset)
    test_robust_accuracy = robust_accuracy_total / len(test_loader.dataset)
    print('Testing Student： natural accuracy:{:.4f},  pgd accuracy:{:.4f}'.format(test_natural_accuracy, test_robust_accuracy))
    return test_natural_accuracy, test_robust_accuracy
def eval_fgsm_whitebox(encoder, classifier, test_loader, num_steps, step_size):
    encoder.eval()
    classifier.eval()
    natural_accuracy_total, robust_accuracy_total = 0, 0
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        natural_accuracy, robust_accuracy = pgd_whitebox(encoder, classifier, X, y, num_steps, step_size)
        natural_accuracy_total += natural_accuracy
        robust_accuracy_total += robust_accuracy
    test_natural_accuracy = natural_accuracy_total / len(test_loader.dataset)
    test_robust_accuracy = robust_accuracy_total / len(test_loader.dataset)
    print('Testing Student： natural accuracy:{:.4f},  fgsm accuracy:{:.4f}'.format(test_natural_accuracy, test_robust_accuracy))
    return test_natural_accuracy, test_robust_accuracy
def main(args):
    model = encoder_linear(num_classes=10).to(device)
    encoder, classifier = model.encoder, model.classifier
    task_mlp = dual_mlp(input_size=512, output_size=10).to(device)
    confidence_mlp = ConfidenceLabelLoss(emb_dim=512, hidden=64, theta=args.theta3, temp=args.temp).to(device)
    cross_fusion = ProtoCrossAttn(D=512, n_head=8).to(device)
    ema_model = copy.deepcopy(model)
    move_average = EMAModel(model=model, ema_model=ema_model, update_bn=True)
    natural_proto_pool = torch.zeros(args.proto_num, args.proto_dim).to(device)
    robust_proto_pool = torch.zeros(args.proto_num, args.proto_dim).to(device)
    all_params = [
        {'params': encoder.parameters(), 'lr': args.lr},
        {'params': classifier.parameters(), 'lr': args.lr},
        {'params': confidence_mlp.parameters(), 'lr': args.lr},
        {'params': task_mlp.parameters(), 'lr': args.lr},
        {'params': cross_fusion.parameters(), 'lr': args.lr},
    ]
    optimizer = optim.SGD(all_params, momentum=args.momentum, weight_decay=args.weight_decay)
    start = time.time()
    unique_label = torch.arange(0, 10)
    prototype_contrastive = PrototypeSupConLoss()
    criterion_nll = nn.NLLLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    logger = Logger(os.path.join(args.model_dir, 'cat.log'))
    logger.log('Using device: {}'.format(device))
    logger.log('\n')
    logger.log('Training for {} epochs'.format(args.epochs))
    logger.log('\n')
    for epoch in range(1, args.epochs + 1):
        logger.log('=============Epoch {}=============='.format(epoch))
        if epoch >= 41:
            for p in classifier.parameters():
                p.requires_grad = False
        if epoch <= 40:
            proto_momentum = args.proto_momentum
            rampup = 0.5
        else:
            proto_momentum = 1.
            rampup = 0.5
        adjust_learning_rate(args, optimizer, epoch)
        train_test_start = time.time()
        print('=============================Training Epoch {}================================='.format(epoch))
        train_natural_accuracy, train_robust_accuracy, natural_proto_pool, robust_proto_pool = train(args, encoder, classifier, move_average,
          cross_fusion, confidence_mlp, task_mlp, train_loader, optimizer, prototype_contrastive, natural_proto_pool, robust_proto_pool, proto_momentum,
          rampup, unique_label, epoch, criterion_nll, criterion_kl)
        train_end = time.time()
        logger.log('Training: Robust Accuracy: {:.4f}.\tNatural Accuracy: {:.4f}.\tTime taken: {:.4f}'
            .format(train_robust_accuracy, train_natural_accuracy, (train_end - train_test_start) / 60))
        print('第[%d]个epoch的训练时间: %.3f' % (epoch, (train_end - train_test_start) / 60) + ' min')
        print('==============================Testing Epoch {}================================'.format(epoch))
        test_natural_accuracy, test_pgd_accuracy = student_pgd_whitebox(encoder, classifier, test_loader, num_steps=args.test_pgd_steps, step_size=args.pgd_step_size)
        test_natural_accuracy, test_fgsm_accuracy = student_fgsm_whitebox(encoder, classifier, test_loader, num_steps=args.test_fgsm_steps, step_size=args.fgsm_step_size)
        test_end = time.time()
        logger.log(
                'Testing Student: Natural Accuracy: {:.4f}.\tPGD Accuracy: {:.4f}\tFGSM Accuracy: {:.4f}\tTime Taken:{:.4f}'
                .format(test_natural_accuracy, test_pgd_accuracy, test_fgsm_accuracy,
                        (test_end - train_test_start) / 60))
        if epoch in [40, 80, 120]:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_{}.py'.format(epoch)))
            torch.save(confidence_mlp.state_dict(), os.path.join(args.model_dir, 'confidence_mlp_{}.py'.format(epoch)))
            torch.save(task_mlp.state_dict(), os.path.join(args.model_dir, 'task_mlp_two_{}.py'.format(epoch)))
            torch.save(cross_fusion.state_dict(), os.path.join(args.model_dir, 'cross_fusion_{}.py'.format(epoch)))
            torch.save({'natural_proto_pool': natural_proto_pool}, os.path.join(args.model_dir, 'natural_proto_pool_{}.mat'.format(epoch)))
            torch.save({'robust_proto_pool': robust_proto_pool}, os.path.join(args.model_dir, 'robust_proto_pool_{}.mat'.format(epoch)))
        else:
            pass
    end = time.time()
    print('最终时间: %.3f' % ((end - start) / 60) + ' min')
    logger.log('\n')
    logger.log('Total time: {}'.format((end - start) / 60))
    logger.log('Script Completed')
    return

if __name__ == '__main__':
    main(args)

