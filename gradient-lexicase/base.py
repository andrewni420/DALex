# Modified from: https://github.com/kuangliu/pytorch-cifar

'''Training image classification on CIFAR-10/100 and SVHN
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from collections import deque


class Net(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

parser = argparse.ArgumentParser(description='Training baseline image classification')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default=0, type=int, help='arch index')
parser.add_argument('--dataset', default='C10', type=str, help='use C10, C100 or SVHN')
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--save', action='store_true', help='save checkpoint')
parser.add_argument('--std', type=float,default=[0.0,0.0], nargs="+", help='lexicase weighting')
parser.add_argument('--batch_size', type=int,default=128, help='batch size')
parser.add_argument('--softmax', action="store_true")
args = parser.parse_args()

torch.random.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0



# Data
print('==> Preparing data..')
if args.dataset == "C10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    num_classes = 10

elif args.dataset == "C100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 100

elif args.dataset == "SVHN":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10

elif args.dataset=="MNIST":
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=64, shuffle=False, num_workers=2)
    num_classes = 10


n_epoch = 200
# lexicase = Lexicase(args.std[0], device=device, softmax=args.softmax)
# Model
print('==> Building model..')
print(args.__dict__)
net = [VGG, ResNet18, ResNet50][args.arch]
net = net(num_classes=num_classes).to(device)
# net = MNISTNet().to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
# criterion = nn.NLLLoss(reduction = )
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

# Training
def train(epoch):
    std = ((epoch-start_epoch)/(n_epoch))*(args.std[0]-args.std[1])+args.std[1]
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        weights = torch.randn(loss.shape,device=device)*std
        loss_=(weights.softmax(0)*loss).sum()

        # loss_ = lexicase(loss, scale=std, softmax=True).mean()
        loss_=loss.mean()
        loss_.backward()
        optimizer.step()

        train_loss += loss.mean().item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    return 100.*correct/total

# training
for epoch in range(start_epoch, start_epoch+n_epoch):
    loss=train(epoch)
    acc = test(epoch)
    scheduler.step()
    

print(args.arch, acc)

# Save checkpoint.
if args.save:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    save_dir = 'ckpt_base_{}_{}_{}_{}'.format(args.dataset, args.arch, args.seed, acc)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, save_dir+'/ckpt.pth')
    print('Checkpoint saved to:', save_dir)


#Normal 
# Epoch: 199
#  [================================================================>]  Step: 11ms | Tot: 5s215ms | Loss: 0.804 | Acc: 71.786% (35893/50000) 391/391 
#  [================================================================>]  Step: 1ms | Tot: 669ms | Loss: 0.801 | Acc: 72.490% (7249/10000) 79/79 

#Lexi0 for FCNs
# Epoch: 199
#  [================================================================>]  Step: 5ms | Tot: 4s979ms | Loss: 0.752 | Acc: 73.740% (36870/50000) 391/391  
#  [================================================================>]  Step: 1ms | Tot: 690ms | Loss: 0.747 | Acc: 74.310% (7431/10000) 79/79 

#Lexi0.1 for FCNs
# Epoch: 199
#  [================================================================>]  Step: 4ms | Tot: 5s709ms | Loss: 0.792 | Acc: 72.134% (36067/50000) 391/391  
#  [================================================================>]  Step: 2ms | Tot: 705ms | Loss: 0.794 | Acc: 72.370% (7237/10000) 79/79 