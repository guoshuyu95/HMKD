import os
import random
import numpy as np
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model_student import VisionTransformer as vit_student
from load_data_multi import read_data, MyDataSet_new


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    device = torch.device("cuda")
    # Data loading
    stain_mode = 'HE'  # it will be replaced by 'IHC' in read data path
    root_path = '/data'
    test_imageA_path, test_imageB_path, test_label = read_data(os.path.join(root_path, fold, 'test', stain_mode))

    # Setting DataSet and dataloader
    transformA = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformB = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = MyDataSet_new(imageA_path=test_imageA_path,
                                 imageB_path=test_imageB_path,
                                 images_class=test_label,
                                 transform=test_transform,
                                 transformA=transformA,
                                 transformB=transformB)

    num_workers = 4
    batch_patch = 16
    num_class = 4

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_patch, shuffle=False,
                                              num_workers=num_workers)

    # create model
    model = vit_student(num_classes=num_class).cuda()

    weights_path = './best_student_f1.pth.tar'
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights_dict['state_dict'])
        print('weighted_f1:', weights_dict['best_weighted_f1'])
    else:
        print('==> Not found training weights pth file.')
        sys.exit(1)

    all_test_logits = []
    all_test_labels = []
    all_test_scores = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        with autocast():
            for idx, (inputA, inputB, target, index, contrast_idx) in enumerate(test_loader):
                imagesA = inputA.cuda()
                target = target.cuda()

                # compute output
                _, logits = model(imagesA)

                all_test_logits.append(logits)
                all_test_labels.append(target)
                all_test_scores.append(torch.softmax(logits, dim=1))

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

    # compute and store classification report
    report = classification_report(all_test_labels, all_test_preds, digits=4, output_dict=True)
    out_path = './classification_report_KD.pkl'
    with open(out_path, 'wb') as file:
        pickle.dump(report, file)


if __name__ == '__main__':
    main()
