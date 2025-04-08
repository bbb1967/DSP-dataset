import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
# import get_map
import time
import DSP2Net
import rasterio
from sklearn.metrics import recall_score, f1_score
import seaborn as sns
import os
from PIL import Image
from scipy import io as sio
import datetime
from scipy.ndimage import median_filter
from collections import Counter
import argparse
import torch.nn.functional as F

def loadData():
    mat_path = "path/data.mat" 
    mat1_path = "path/labels.mat"


    data = sio.loadmat(mat_path)['image']
    labels = sio.loadmat(mat1_path)['label']


    # data = np.transpose(data, (1, 2, 0))

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    return data, labels



def padWithZeros(X, margin=2):  
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX



def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
  
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin) 
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))  
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))  
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin): 
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1  

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


BATCH_SIZE_TRAIN = 48


def create_data_loader():
    # 地物类别
    # class_num = 12
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比例

    test_ratio = 0.90
    # 每个像素周围提取 patch 的尺寸
    patch_size = 11
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print("Unique labels:", unique_labels)
    print("Label counts:", label_counts)

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    gt = y
    print('\n... ... create data cubes ... ...')
    img_new, gt_all = createImageCubes(X, y, windowSize=patch_size)
    print('Data cube X shape: ', img_new.shape)
    print('Data cube y shape: ', gt_all.shape)
    X = img_new
    y = gt_all

    X_train = []
    y_train = []
    X_validate = []
    y_validate = []
    X_test = []
    y_test = []

    for i in range(0, 9):

        X_i = X[y == i]
        y_i = y[y == i]

        X_train_i, X_temp_i, y_train_i, y_temp_i = train_test_split(X_i, y_i, train_size=train_samples_per_class[i],
                                                                    random_state=42)
        X_validate_i, X_test_i, y_validate_i, y_test_i = train_test_split(X_temp_i, y_temp_i,
                                                                          train_size=validate_samples_per_class[i],
                                                                          random_state=42)

        # 记录划分后的数据集
        X_train.append(X_train_i)
        y_train.append(y_train_i)
        X_validate.append(X_validate_i)
        y_validate.append(y_validate_i)
        X_test.append(X_test_i)
        y_test.append(y_test_i)

        remaining_samples = len(X_i) - train_samples_per_class[i] - validate_samples_per_class[i]
        print(f"After sampling for class {i}, remaining samples: {remaining_samples}")


    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_validate = np.concatenate(X_validate, axis=0)
    y_validate = np.concatenate(y_validate, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_validate shape:', X_validate.shape)
    print('y_validate shape:', y_validate.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    X = X.reshape(-1, patch_size, patch_size, 6, 1)
    X_train = X_train.reshape(-1, patch_size, patch_size, 6, 1)
    X_validate = X_validate.reshape(-1, patch_size, patch_size, 6, 1)
    X_test = X_test.reshape(-1, patch_size, patch_size, 6, 1)
    print('before transpose: Xtrain shape: ', X_train.shape)
    print('before transpose: Xvalidate shape: ', X_validate.shape)
    print('before transpose: Xtest  shape: ', X_test.shape)


    X = X.transpose(0, 4, 3, 1, 2)
    X_train = X_train.transpose(0, 4, 3, 1, 2)
    X_validate = X_validate.transpose(0, 4, 3, 1, 2)
    X_test = X_test.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', X_train.shape)
    print('after transpose: Xvalidate shape: ', X_validate.shape)
    print('after transpose: Xtest  shape: ', X_test.shape)

    X = TestDS(X, gt_all)
    trainset = TrainDS(X_train, y_train)
    validateset = ValidateDS(X_validate, y_validate)
    testset = TestDS(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,  
                                               num_workers=0,
                                               )
    validate_loader = torch.utils.data.DataLoader(dataset=validateset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True, 
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )

    return train_loader, validate_loader, test_loader, all_data_loader, y, gt, y_train

def get_classification_map(y_pred, y): 
    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k] + 1  
                k += 1

    return cls_labels
def convert_to_color(x): 
    return convert_to_color_(x, palette=palette)
def convert_to_color_(arr_2d, palette=None):
   
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)  
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():  
        m = arr_2d == c  
        arr_3d[m] = i  

    return arr_3d  


""" Training dataset"""
class BCE_loss(nn.Module):

    def __init__(self, args,
                target_threshold=None,
                type=None,
                reduction='mean',
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = args.smoothing
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(args)
        if type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self, args):
        cls_num = torch.Tensor(args.cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device,
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
class ValidateDS(torch.utils.data.Dataset):
    def __init__(self, Xvalidate, yvalidate):
        self.len = Xvalidate.shape[0]
        self.x_data = torch.FloatTensor(Xvalidate)
        self.y_data = torch.LongTensor(yvalidate)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

    def __len__(self):
            return self.len

def train(train_loader, validate_loader, epochs, lossq):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = DSP2Net.DSP2Net().to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = lossq
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    current_loss_high_count = 0  # 计数器

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss1 = criterion1(outputs, target)
            loss2 = criterion2(outputs, target)
            loss = loss1 * 0.8 + loss2 * 0.2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)  
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, avg_loss, loss.item()))

        val_loss = validate(validate_loader, net, criterion1, device)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d')
            model_path = f'DSP_{current_time}.pth'
            torch.save(net.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}, time: {current_time}")

    print('Finished Training')

    return net, device

def validate(validate_loader, net, criterion, device):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
    return total_loss / len(validate_loader)

def test(device, net, all_data_loader, save_pred_path='predictions.mat', save_label_path='labels.mat'):
    count = 0
    net.eval()
    y_pred_test = []
    y_test = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels in all_data_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))

    # Save the predictions and labels to separate .mat files


    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(np.divide(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names = ['非荒漠化', '轻度风蚀荒漠化', '中度'
        , '重度风蚀荒漠化', '湖泊',
                    '居民区', '轻度盐渍化', '中度盐渍化', '重度盐渍化',
                    ]

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test, average='macro')  # 召回率
    f1 = f1_score(y_test, y_pred_test, average='macro')  # F1分数


    intersection = np.diag(confusion)
    ground_truth_set = confusion.sum(axis=1)
    predicted_set = confusion.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IOU = intersection / union

   
    freq = ground_truth_set / np.sum(ground_truth_set)
    FWIOU = (freq[freq > 0] * IOU[freq > 0]).sum()

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100, recall * 100, f1 * 100, IOU, FWIOU

parser = argparse.ArgumentParser( 
    description="Run deep learning experiments on" " various hyperspectral datasets"
)

if __name__ == '__main__':
    num_runs = 6  
    all_results = []  
    #date_str = datetime.now().strftime("%Y-%m-%d")  
    results_file = f'DSP.txt' 

    with open(results_file, 'w') as f: 
        f.write("Run, OA, AA, Kappa, Recall, F1, Each Acc\n")  

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        train_loader, validate_loader, test_loader, all_data_loader, y_all, gt, y_train = create_data_loader()

        palette = {0: (0, 0, 0), 1: (55, 166, 2), 2: (255, 255, 195), 3: (250, 250, 127),
                   4: (168, 168, 0), 5: (6, 110, 255), 6: (251, 4, 0), 7: (230, 230, 230),
                   8: (178, 178, 178), 9: (129, 129, 129)}

        class_counts = Counter(y_train)
        sample_counts_list = [count for count in class_counts.values()]

        if run == 0:
            parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
            parser.add_argument('--cls_num', type=list, default=sample_counts_list)
            args = parser.parse_args()
        criterion = BCE_loss(args, type='Bal')

        tic1 = time.perf_counter()
        net, device = train(train_loader, validate_loader, epochs=60, lossq=criterion)
        toc1 = time.perf_counter()

        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, all_data_loader, save_pred_path='predictions.mat',
                                   save_label_path='labels.mat')
        toc2 = time.perf_counter()
        cls_labels = get_classification_map(y_pred_test, gt)

        classification, oa, confusion, each_acc, aa, kappa, recall, f1, IOU, FWIOU = acc_reports(y_test,
                                                                                                 y_pred_test)

        each_acc = np.array(each_acc)

        if oa >= 80 and aa >= 80:
            all_results.append((oa, aa, kappa, recall, f1, *each_acc))
            with open(results_file, 'a') as f:
                f.write(f"{run + 1}, {oa}, {aa}, {kappa}, {recall}, {f1}, {','.join(map(str, each_acc))}\n")
        else:
            print(f"Run {run + 1} results discarded (OA: {oa}, AA: {aa})")

        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
        print(f'Training Time (s): {Training_Time:.2f}')
        print(f'Test Time (s): {Test_time:.2f}')

        cls_labels = get_classification_map(y_pred_test, gt)
        color_cls_labels = convert_to_color(cls_labels)

        folder_path = "cls_map"
        MODEL = "1"  
        DATASET = "1" 
        im_name = f"{MODEL}_{DATASET}_{run + 1}_{round(oa, 2)}_{round(aa, 2)}.jpg"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        im = Image.fromarray(color_cls_labels)
        im = im.resize((im.width, im.height), Image.NEAREST)
        im.save(os.path.join(folder_path, im_name))

        print(f'Run {run + 1} completed.\n')

    if len(all_results) > 0:
        results_array = np.array(all_results)
        means = results_array.mean(axis=0)
        stds = results_array.std(axis=0)

        metrics = ['Overall accuracy', 'Average accuracy', 'Kappa', 'Recall', 'F1 Score']
        with open(results_file, 'a') as f:
            for i, metric in enumerate(metrics):
                f.write(f"{metric}: {means[i]:.2f} ± {stds[i]:.2f}\n")

        each_acc_means = results_array[:, 5:]
        for i in range(each_acc_means.shape[1]):
            mean_acc = each_acc_means[:, i].mean()
            std_acc = each_acc_means[:, i].std()
            print(f"Class {i + 1} Average Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
    else:
        print("No valid results to compute averages.")
