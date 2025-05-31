import os
import random
import shutil
import time
from enum import Enum
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
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

import torch.nn.functional as F
from model_teacher import VisionTransformer as vit
from model_student import VisionTransformer as vit_student
from load_data_multi import read_data, MyDataSet_new
from loss_attn import AttnLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def main(start_epoch=0):
    setup_seed(40)

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
    num_workers = 4
    batch_patch = 16
    lr = 0.001
    kd_T = 5

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

    # define teacher model
    model_t = vit(num_classes=num_class).cuda()
    weights_path = './best_teacher_f1.pth.tar'
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        model_t.load_state_dict(weights_dict['state_dict'])
        print('==> Best training weights for teacher model have been loaded!')
        print('best_weighted_f1:', weights_dict['best_weighted_f1'])
        print('best_acc:', weights_dict['best_acc'])
    else:
        print('==> Not found training weights pth file.')
        sys.exit(1)

    # define student model
    model_s = vit_student(num_classes=num_class).cuda()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_attn0 = AttnLoss()
    criterion_attn1 = AttnLoss()
    criterion_attn2 = AttnLoss()
    criterion_kd3 = torch.nn.CosineEmbeddingLoss(reduction='sum')

    print('======================================')

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_attn0)
    criterion_list.append(criterion_attn1)
    criterion_list.append(criterion_attn2)
    criterion_list.append(criterion_kd3)

    # optimizer
    optimizer = torch.optim.SGD(model_s.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)
    criterion_list.cuda()

    scaler = GradScaler()

    for epoch in range(start_epoch, epoches):
        print("learning rate:", scheduler.get_last_lr())
        # train for one epoch
        model_s = train(train_loader, model_t, model_s, optimizer, criterion_list, epoch, scaler)
        # evaluate on validation set
        test_acc, test_weighted_f1 = validate(test_loader, model_s, criterion_cls)

        scheduler.step()

        # record best f1 and acc, save checkpoint
        is_best_f1 = test_weighted_f1 > best_weighted_f1
        best_weighted_f1 = max(test_weighted_f1, best_weighted_f1)

        if is_best_f1:
            final_acc = test_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_s.state_dict(),
            'best_weighted_f1': best_weighted_f1,
            'best_acc': final_acc
        }, is_best_f1=is_best_f1)

        print('best_weighted_f1', best_weighted_f1)


def train(train_loader, model_t, model_s, optimizer, criterion_list, epoch, scaler, print_freq=500):
    Loss = AverageMeter('Loss', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [Loss],
        prefix="Epoch: [{}]".format(epoch))

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_attn0 = criterion_list[2]
    criterion_attn1 = criterion_list[3]
    criterion_attn2 = criterion_list[4]
    criterion_kd3 = criterion_list[5]

    model_s.train()
    model_t.eval()

    start = time.time()
    for i, (inputA, inputB, target) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = inputA.size(0)
        # A is HE, B is IHC
        inputA = inputA.cuda()
        inputB = inputB.cuda()
        target = target.cuda()

        # ===================forward=====================
        with torch.no_grad():
            with autocast():
                # model teacher
                feat_t, logit_t = model_t(inputA, inputB)
                feat_t = [f.detach() for f in feat_t]
                attn_a0_t, attn_a1_t, attn_a2_t = model_t.attn_a0, model_t.attn_a1, model_t.attn_a2
                attn_a0_t, attn_a1_t, attn_a2_t = attn_a0_t.detach(), attn_a1_t.detach(), attn_a2_t.detach()

        with autocast():
            feat_s, logit_s = model_s(inputA)
            attn_a0_s, attn_a1_s, attn_a2_s = model_s.attn_a0, model_s.attn_a1, model_s.attn_a2

            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            # hmkd
            loss_f0 = criterion_attn0(attn_a0_s, attn_a0_t)
            loss_f1 = criterion_attn1(attn_a1_s, attn_a1_t)
            loss_f2 = criterion_attn2(attn_a2_s, attn_a2_t)
            loss_f3 = criterion_kd3(feat_s[3], feat_t[3], torch.ones([batch_size]).cuda())

        a = 1.0
        b = 1.0
        loss = loss_cls + loss_div + a * loss_f3 + b * (loss_f0 + loss_f1 + loss_f2)

        Loss.update(loss.item(), inputA.size(0))

        # ===================backward=====================
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % print_freq == 0:
            progress.display(i + 1)

    end = time.time()

    print('Epoch Finished')
    print('==> Epoch [{}] cost Time: {:.2f}s'.format(epoch, (end - start)))

    return model_s


def validate(test_loader, model_s, criterion_cls, print_freq=200):
    Loss = AverageMeter('=>Loss:', ':.6f')
    progress = ProgressMeter(
        len(test_loader),
        [Loss],
        prefix='Test: ')

    all_test_logits = []
    all_test_labels = []
    # switch to evaluate mode
    model_s.eval()
    with torch.no_grad():
        with autocast():
            for i, (imageA, imageB, target) in enumerate(test_loader):
                imagesA = imageA.cuda()
                target = target.cuda()

                # compute output
                _, out_s_logit = model_s(imagesA)
                loss = criterion_cls(out_s_logit, target)

                Loss.update(loss, imagesA.size(0))

                all_test_logits.append(out_s_logit)
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


def save_checkpoint(state, is_best_f1, filename='./checkpoint_s.pth.tar'):
    torch.save(state, filename)
    if is_best_f1:
        shutil.copyfile(filename, './best_KD_f1.pth.tar')


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
