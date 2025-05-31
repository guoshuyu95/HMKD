import os
import random
import shutil
import time
from enum import Enum
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score

from model_teacher import VisionTransformer as vit
from load_data_multi import read_data, MyDataSet_new


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(start_epoch=0):
    setup_seed(42)

    device = torch.device("cuda")

    # Data loading
    stain_mode = 'HE'  # it will be replaced by 'IHC' in read data path
    root_path = './data'

    # A is HE, B is IHC
    train_imageA_path, train_imageB_path, train_label = read_data(os.path.join(root_path, 'train', stain_mode))
    test_imageA_path, test_imageB_path, test_label = read_data(os.path.join(root_path, 'test', stain_mode))

    # Setting DataSet and dataloader
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip()])
    transformA = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformB = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MyDataSet_new(imageA_path=train_imageA_path,
                                  imageB_path=train_imageB_path,
                                  images_class=train_label,
                                  transform=train_transform,
                                  transformA=transformA,
                                  transformB=transformB)
    test_dataset = MyDataSet_new(imageA_path=test_imageA_path,
                                 imageB_path=test_imageB_path,
                                 images_class=test_label,
                                 transform=test_transform,
                                 transformA=transformA,
                                 transformB=transformB)

    epoches = 30
    num_workers = 8
    batch_patch = 16
    lr = 0.001

    best_weighted_f1 = 0
    final_acc = 0
    num_class = 4

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_patch, shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_patch, shuffle=False,
                                              num_workers=num_workers)

    # create model
    model = vit(num_classes=num_class).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)
    scaler = GradScaler()

    for epoch in range(start_epoch, epoches):
        print("**learning rate:", scheduler.get_last_lr())
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scaler)
        # evaluate on validation set
        test_acc, test_weighted_f1 = validate(test_loader, model, criterion)

        scheduler.step()

        # record best f1 and acc, save checkpoint
        is_best_f1 = test_weighted_f1 > best_weighted_f1
        best_weighted_f1 = max(test_weighted_f1, best_weighted_f1)

        if is_best_f1:
            final_acc = test_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_weighted_f1': best_weighted_f1,
            'best_acc': final_acc
        }, is_best_f1=is_best_f1)

        print('best_weighted_f1', best_weighted_f1)


def train(train_loader, model, criterion, optimizer, epoch, scaler, print_freq=500):
    Loss = AverageMeter('Loss', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [Loss],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    start = time.time()
    for i, (imageA, imageB, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # move data to the same device as model
        imageA = imageA.cuda()
        imageB = imageB.cuda()
        labels = labels.cuda()

        # ===================forward=====================
        with autocast():
            _, logits = model(imageA, imageB)
            loss = criterion(logits, labels)

        Loss.update(loss, imageA.size(0))

        # ===================backward=====================
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % print_freq == 0:
            progress.display(i + 1)

    end = time.time()

    print('Epoch Finished')
    print('==> Epoch [{}] cost Time: {:.2f}s'.format(epoch, (end - start)))


def validate(test_loader, model, criterion, print_freq=200):
    Loss = AverageMeter('=>Loss cls:', ':.6f')
    progress = ProgressMeter(
        len(test_loader),
        [Loss],
        prefix='Test: ')

    all_test_logits = []
    all_test_labels = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with autocast():
            for i, (imageA, imageB, target) in enumerate(test_loader):
                imageA = imageA.cuda()
                imageB = imageB.cuda()
                target = target.cuda()

                # compute output
                _, logits = model(imageA, imageB)
                loss = criterion(logits, target)
                # record
                Loss.update(loss, imageA.size(0))

                all_test_logits.append(logits)
                all_test_labels.append(target)

                if i % print_freq == 0:
                    progress.display(i + 1)
    progress.display_summary()
    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = torch.argmax(all_test_logits, dim=1)
    all_test_labels = torch.cat(all_test_labels).cpu()

    # compute accuracy and weighted f1-score
    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')

    print('=> Test Acc:', accuracy)
    print('=> Test Weighted F1:', weighted_f1_score)
    return accuracy, weighted_f1_score


def save_checkpoint(state, is_best_f1, filename='./checkpoint_.pth.tar'):
    torch.save(state, filename)
    if is_best_f1:
        shutil.copyfile(filename, './best_teacher_f1.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
