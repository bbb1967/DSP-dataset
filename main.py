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


def loadData():
    mat_path = "path/data"  # 多光谱影像.mat文件路径
    mat1_path = "path/label"

    # 从.mat文件中加载数据
    data = sio.loadmat(mat_path)['image']
    labels = sio.loadmat(mat1_path)['label']

    #data = data[:, 0:1500, :]
    #labels = labels[:, 0:1500]
    #data = np.transpose(data, (1, 2, 0))
    # 打印形状
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    return data, labels



# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):  # 0填充，填充的边距margin
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取patch，然后创建成符合keras处理的格式，并返回相对应的标签
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给X做padding，removeZeroLabels表示是否移除标签为零的图像块（默认值为True）
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)  # 进行0填充，以提取边缘图像块
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))  # 用于存储图像块
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))  # 存储相应图像块中心像素的标签
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):  # 遍历X中的每个元素
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1  # 可能是为了调整标签值的范围，使得类别索引从0开始

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
    patch_size = 9
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

    # 假设 num_train_samples_per_class 是一个字典，存储每个类别需要的训练样本数量
    train_samples_per_class = {0: 35000, 1: 25100, 2: 15200, 3: 12000,
                               4: 10000, 5: 8000, 6: 9200, 7: 8500, 8: 5000}
    validate_samples_per_class = {0: 1120, 1: 730, 2: 114, 3: 30,
                                  4: 424, 5: 230, 6: 40, 7: 5, 8: 13}  #

    # 初始化训练集和测试集的列表
    X_train = []
    y_train = []
    X_validate = []
    y_validate = []
    X_test = []
    y_test = []

    # 对每个类别进行循环
    for i in range(0, 9):
        # 提取该类别的数据和标签
        X_i = X[y == i]
        y_i = y[y == i]
        # 划分训练集和测试集，按照指定的训练样本数量  random_state用来设置一个随机取样的格式，可以重现结果
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

    # 将列表转换为数组
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_validate = np.concatenate(X_validate, axis=0)
    y_validate = np.concatenate(y_validate, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # 打印训练集和测试集的形状
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_validate shape:', X_validate.shape)
    print('y_validate shape:', y_validate.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X.reshape(-1, patch_size, patch_size, 6, 1)
    X_train = X_train.reshape(-1, patch_size, patch_size, 6, 1)
    X_validate = X_validate.reshape(-1, patch_size, patch_size, 6, 1)
    X_test = X_test.reshape(-1, patch_size, patch_size, 6, 1)
    print('before transpose: Xtrain shape: ', X_train.shape)
    print('before transpose: Xvalidate shape: ', X_validate.shape)
    print('before transpose: Xtest  shape: ', X_test.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    X_train = X_train.transpose(0, 4, 3, 1, 2)
    X_validate = X_validate.transpose(0, 4, 3, 1, 2)
    X_test = X_test.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', X_train.shape)
    print('after transpose: Xvalidate shape: ', X_validate.shape)
    print('after transpose: Xtest  shape: ', X_test.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, gt_all)
    trainset = TrainDS(X_train, y_train)
    validateset = ValidateDS(X_validate, y_validate)
    testset = TestDS(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,   # 这行代码设置了是否在每个训练周期开始时打乱数据的顺序，这里设置为True，表示需要打乱
                                               num_workers=0,
                                               )
    validate_loader = torch.utils.data.DataLoader(dataset=validateset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,  # 这行代码设置了是否在每个训练周期开始时打乱数据的顺序，这里设置为True，表示需要打乱
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

    return train_loader, validate_loader, test_loader, all_data_loader, y, gt

def get_classification_map(y_pred, y):  # 将模型的分类预测(y_pred)映射到一个大小与真实标签(y)相同的二维数组

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
                cls_labels[i][j] = y_pred[k] + 1  # patchesLabels-=1相对应，即类别标签从0开始的，此时赋值时要加上1
                k += 1

    return cls_labels
def convert_to_color(x):  # 将标签数组转换为RGB颜色编码的图像
    return convert_to_color_(x, palette=palette)
def convert_to_color_(arr_2d, palette=None):
    """ 将标签数组转换为RGB颜色编码的图像.

    参数:
        arr_2d：包含标签的整数2D数组
        palette：颜色字典，用于表示标签号与RGB元组的对应关系

    返回:
        arr_3d：使用RGB格式编码的颜色标签的整数2D图像

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)  # 创建一个全零的三维数组，表示图像
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():  # 对颜色调色板中的每个标签号和相应的RGB元组进行迭代
        m = arr_2d == c  # 创建一个布尔掩码，其中标签数组中等于当前标签号c的位置为True，其他位置为False
        arr_3d[m] = i  # 使用布尔掩码将对应位置的三维数组元素赋值为当前标签号c对应的RGB元组i

    return arr_3d  # 返回转换后的RGB彩色图像数组


""" Training dataset"""


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

def train(train_loader, validate_loader, epochs):
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = DSP2Net.DSP2Net().to(device)
    #class_weights = torch.tensor([1.6, 4, 15, 6, 6.3, 6.8, 6.2, 10, 25], dtype=torch.float).to(device)

    # 将权重传递给 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # 开始训练
    for epoch in range(epochs):
        net.train()
        total_loss = 0   # 每个周期损失初始化为0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()   # 更新网络的参数
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),  # 平均损失
                                                                         loss.item()))  # 当前损失  # 打印平均损失

        # 在验证集上计算损失并调整学习率
        val_loss = validate(validate_loader, net,criterion, device)
        scheduler.step(val_loss)  # 根据验证集的损失值更新学习率
        if (epoch + 1) % 20 == 0:  # 每隔20个epoch保存一次模型
            # 获取当前时间点并格式化
            current_time = datetime.datetime.now().strftime('%Y-%m-%d')
            # 设置模型保存路径和文件名
            model_path = f'_{current_time}.pth'
            # 保存模型参数
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
    target_names = ['1', '2', '3'
        , '4', '5',
                    '6', '7', '8', '9',
                    ]

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test, average='macro')  # 召回率
    f1 = f1_score(y_test, y_pred_test, average='macro')  # F1分数

    # 计算交并比（IOU）
    intersection = np.diag(confusion)
    ground_truth_set = confusion.sum(axis=1)
    predicted_set = confusion.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IOU = intersection / union

    # 计算频率加权交并比（FWIOU）
    freq = ground_truth_set / np.sum(ground_truth_set)
    FWIOU = (freq[freq > 0] * IOU[freq > 0]).sum()

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100, recall * 100, f1 * 100, IOU, FWIOU


# 在主函数中调用 acc_reports 函数并打印结果
if __name__ == '__main__':
    train_loader, validate_loader, test_loader, all_data_loader, y_all, gt = create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, validate_loader, epochs=60)
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, all_data_loader, save_pred_path='predictions.mat',
                               save_label_path='labels.mat')
    toc2 = time.perf_counter()
    classification, oa, confusion, each_acc, aa, kappa, recall, f1, IOU, FWIOU = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    label_values = [
        "Undefined",
        "非荒漠化",
        "轻度风蚀荒漠化",
        "中度风蚀荒漠化",
        "重度风蚀荒漠化",
        "湖泊",
        "居民区",
        "轻度盐渍化",
        "中度盐渍化",
        "重度盐渍化",
        # "盐土",
    ]
    palette = {
        0: (0, 0, 0),  # 类别0的颜色 (白色)
        1: (55, 166, 2),  # 类别0的颜色 (白色)
        2: (255, 255, 195),  # 类别1的颜色 (红色)
        3: (250, 250, 127),  # 类别2的颜色 (蓝色)
        #   3: (255, 255, 4),    # 类别3的颜色 (绿色)

        4: (168, 168, 0),  # 类别4的颜色 (黄色)
        5: (6, 110, 255),  # 类别5的颜色 (紫色)
        6: (251, 4, 0),  # 类别6的颜色 (粉色)
        7: (230, 230, 230),  # 类别7的颜色 (青色)
        # 7: (204, 204, 204),
        8: (178, 178, 178),
        9: (129, 129, 129),
    }

    # 可选：打印调色板以验证分配的颜色
    for k, color in palette.items():
        print(f"Label {k}: Color {color}")
    y_pred_test, y_new = test(device, net, all_data_loader)
    print(y_pred_test.shape)
    cls_labels = get_classification_map(y_pred_test, gt)

    color_cls_labels = convert_to_color(cls_labels)
    folder_path = "cls_map"
    MODEL = "mode"  # 示例模型名称
    DATASET = "1"  # 示例数据集名称
    run = 1  # 示例运行编号
    rounded_value = round(oa, 2)
    rounded_value_2 = round(aa, 2)
    # 创建文件夹（如果不存在）
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    im = Image.fromarray(color_cls_labels)
    im_name = "{}_{}_{}_{}_{}.jpg".format(MODEL, DATASET, run, rounded_value, rounded_value_2)
    im = im.resize((im.width, im.height), Image.NEAREST)
    im.save(os.path.join(folder_path, im_name))
    print('Training_Time (s):', Training_Time)
    print('Test_time  (s):', Test_time)
    print('Kappa accuracy (%):', kappa)
    print('Overall accuracy (%):', oa)
    print('Average accuracy (%):', aa)
    print('Each accuracy (%):', each_acc)
    print('Recall (%):', recall)
    print('F1 Score (%):', f1)
    print('Classification Report:')
    print(classification)
    print('Confusion Matrix:')
    print(confusion)
    print('IOU:', IOU)
    print('FWIOU:', FWIOU)

