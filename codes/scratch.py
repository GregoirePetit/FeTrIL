#!/usr/bin/env python
# coding=utf-8
from collections import defaultdict
from datetime import timedelta
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
import numpy as np
import time
import os
import sys
from torch.optim.optimizer import Optimizer

from AverageMeter import AverageMeter
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'utilsCIL'))
sys.path.insert(0,os.path.join(os.path.expanduser('~'),'FeTrIL'))

from MyImageFolder import ImagesListFileFolder
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle

import lucir_models.modified_resnet as modified_resnet

######### Modifiable Settings ##########
from configparser import ConfigParser
from Utils import DataUtils
import warnings, socket, os

#concatenate the paths with the true classes
def merge_images_labels(images, labels):
    images = list(images)
    labels = list(labels)
    assert(len(images)==len(labels))
    imgs = []
    for i in range(len(images)):
        item = (images[i], labels[i])
        imgs.append(item)
    
    return imgs


if len(sys.argv) != 2:
    print('Arguments: config')
    sys.exit(-1)

cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)

cp = cp['config']
nb_classes = int(cp['nb_classes'])
normalization_dataset_name = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
il_states = int(cp["il_states"])
feat_root = cp["feat_root"]
list_root = cp["list_root"]
model_root = cp["model_root"]
random_seed = int(cp["random_seed"])
num_workers = int(cp['num_workers'])
epochs_lucir = int(cp['epochs_lucir'])
epochs_augmix_ft = int(cp['epochs_augmix_ft'])

B = first_batch_size
datasets_mean_std_file_path = cp["mean_std"]
output_dir = os.path.join(model_root,normalization_dataset_name,"seed"+str(random_seed),"b"+str(first_batch_size))
train_file_path = os.path.join(list_root,normalization_dataset_name,"train.lst")
test_file_path = os.path.join(list_root,normalization_dataset_name,"test.lst")

utils = DataUtils()
train_batch_size       = 128
test_batch_size        = 50
eval_batch_size        = 128
base_lr                = 0.1
lr_strat               = [30, 60]
lr_factor              = 0.1
custom_weight_decay    = 0.0001
custom_momentum        = 0.9

epochs = epochs_lucir

print("Running on " + str(socket.gethostname()) + " | " + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_mean, dataset_std = utils.get_dataset_mean_std(normalization_dataset_name, datasets_mean_std_file_path)
normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
top = min(5, B)

if False:
    trainset = ImagesListFileFolder(
                train_file_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]), random_seed=random_seed, range_classes=range(B))

    testset = ImagesListFileFolder(
                test_file_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]), random_seed=random_seed, range_classes=range(B))

    ################################
    X_train_total, Y_train_total = np.array(trainset.imgs), np.array(trainset.targets)
    X_valid_total, Y_valid_total = np.array(testset.imgs), np.array(testset.targets)

    # the order is already shuffled by our custom loader
    order_list = list(range(B))

    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []



    #init model
    ############################################################
    tg_model = modified_resnet.resnet18(num_classes=B)
    in_features = tg_model.fc.in_features
    out_features = tg_model.fc.out_features
    print("in_features:", in_features, "out_features:", out_features)


    # Prepare the training data for the current batch of classes
    X_train          = X_train_total
    X_valid          = X_valid_total
    X_valid_cumuls.append(X_valid)
    X_train_cumuls.append(X_train)
    X_valid_cumul    = np.concatenate(X_valid_cumuls)
    X_train_cumul    = np.concatenate(X_train_cumuls)

    Y_train          = Y_train_total
    Y_valid          = Y_valid_total
    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
    Y_train_cumul    = np.concatenate(Y_train_cumuls)

    # Add the stored exemplars to the training data
    X_valid_ori = X_valid
    Y_valid_ori = Y_valid

    # Launch the training loop
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])


    ############################################################
    current_train_imgs = merge_images_labels(X_train, map_Y_train)
    trainset.imgs = trainset.samples = current_train_imgs
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    print('Training-set size = ' + str(len(trainset)))

    current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
    testset.imgs = testset.samples = current_test_imgs
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
        shuffle=False, num_workers=num_workers)
    print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
    ##############################################################
    ckp_name = os.path.join(output_dir,'lucir_scratch.pth')
    print('ckp_name', ckp_name)
    ###############################
    tg_params = tg_model.parameters()
    ###############################
    tg_model = tg_model.to(device)
    tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
    ###############################
    top = min(5, B)
    for epoch in range(epochs):
        tg_model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            tg_optimizer.step()
        tg_lr_scheduler.step()

        # eval
        top1 = AverageMeter()
        top5 = AverageMeter()
        tg_model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
            epoch+1, epochs,  len(testloader), top1.avg, top, top5.avg))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    torch.save(tg_model.state_dict(), ckp_name)
ckp_name = os.path.join(output_dir,'lucir_scratch.pth')
# now we need to finetune the model with the augmix
epochs = epochs_augmix_ft
batch_size=64
lr=0.01
momentum = 0.9
weight_decay=0.0001
lrd=10
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

if device is not None:
    print("Use GPU: {} for training".format(device))

# instantiate a ResNet18 model
model = models.resnet18()
tg_model_state_dict = torch.load(ckp_name)
print("Loading model from {}".format(ckp_name))
state_dict = tg_model_state_dict
# remove the fc layer of the dict tg_model_state_dict
for key in list(state_dict.keys()):
    if key.startswith('fc'):
        del state_dict[key]
model.fc = nn.Linear(512, B)
model.load_state_dict(state_dict, strict=False)



print('modele charge')

model.cuda(device)



# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
optimizer = Lookahead(optimizer)

train_dataset = ImagesListFileFolder(
            train_file_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AugMix(severity=5,chain_depth=7),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(B))

val_dataset = ImagesListFileFolder(
            test_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(B))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=False)

print('Classes number = {}'.format(len(train_dataset.classes)))
print('Training dataset size = {}'.format(len(train_dataset)))
print('Validation dataset size = {}'.format(len(val_dataset)))

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch==80 or epoch==120:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print('\nstarting training...')
start = time.time()
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, lrd)

    # train for one epoch
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):

        if device is not None:
            input = input.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure elapsed time
    end = time.time()
    epoch_time =  timedelta(seconds=round(end - start))

    print('Train | Epoch: [{0}]\t'
        'Time {epoch_time}\t'
          'Loss {loss.avg:.4f}\t'
          'Acc@1 {top1.avg:.3f}\t'
          'Acc@5 {top5.avg:.3f}'.format(
        epoch, epoch_time=epoch_time, loss=losses, top1=top1, top5=top5))
    # evaluate on validation set

    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prec1, prec5 = utils.accuracy(outputs.data, targets, topk=(1, top))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    print('{:03}/{:03} | Test ({}) |  acc@1 = {:.2f} | acc@{} = {:.2f}'.format(
        epoch+1, epochs,  len(val_loader), top1.avg, top, top5.avg))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    
ckp_name = os.path.join(output_dir,'scratch.pth')
torch.save(model, ckp_name)

def features_extraction(features_model, loader, root_path, gpu):
    features_model = features_model.cuda(gpu)
    features_model.eval()
    try:
        print('cleaning',root_path,'...')
        shutil.rmtree(root_path)
    except:
        pass
    os.makedirs(root_path, exist_ok=True)
    last_class = -1
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.cuda(gpu)
        features = features_model(inputs)
        lablist=labels.tolist()
        featlist=features.tolist()
        for i in range(len(lablist)):
            cu_class = lablist[i]
            if cu_class!=last_class:
                last_class=cu_class
            with open(os.path.join(root_path,str(cu_class)), 'a') as features_out:
                features_out.write(str(' '.join([str(e[0][0]) for e in list(featlist[i])])) + '\n')


# now we extract features from the model

train_dataset = ImagesListFileFolder(
            train_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(nb_classes))
val_dataset = ImagesListFileFolder(
            test_file_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), random_seed=random_seed, range_classes=range(nb_classes))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=num_workers, pin_memory=False, shuffle=False)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=1, num_workers=num_workers, pin_memory=False, shuffle=False)

train_features_path = os.path.join(output_dir,'train')
feat_dir_full = os.path.join(feat_root,normalization_dataset_name,"seed"+str(random_seed),"b"+str(first_batch_size))
train_features_path = os.path.join(feat_dir_full,'train')
val_features_path = os.path.join(feat_dir_full,'test')
feat_model = nn.Sequential(*list(model.children())[:-1])
features_extraction(feat_model, train_loader, train_features_path, device)
features_extraction(feat_model, val_loader, val_features_path, device)
