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
import random
from einops import rearrange
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

def loadData():
    mat_path = "path/data"  # 多光谱影像.mat文件路径
    mat1_path = "path/label"

    # 从.mat文件中加载数据
    data = sio.loadmat(mat_path)['image']
    labels = sio.loadmat(mat1_path)['label']

    #data = data[10:3440, :, :]
    #labels = labels[10:3440, :]
    #data = np.transpose(data, (1, 2, 0))
    # 打印形状
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    return data, labels



# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):    # 0填充，填充的边距margin
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX



# 在每个像素周围提取patch，然后创建成符合keras处理的格式，并返回相对应的标签
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给X做padding，removeZeroLabels表示是否移除标签为零的图像块（默认值为True）
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)   # 进行0填充，以提取边缘图像块
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))  # 用于存储图像块
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))   # 存储相应图像块中心像素的标签
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):   # 遍历X中的每个元素
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1   # 可能是为了调整标签值的范围，使得类别索引从0开始

    return patchesData, patchesLabels
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 48

class SpatialContextAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialContextAdapter, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

    def forward(self, x):
        spatial_features = self.conv(x)
        global_context = self.global_pool(spatial_features)
        return spatial_features + global_context.expand_as(spatial_features)


class SemanticResponseAdapter(nn.Module):
    def __init__(self, in_channels, expansion_factor=4):
        super(SemanticResponseAdapter, self).__init__()
        expanded_channels = in_channels * expansion_factor

        self.inverted_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),  # 通道扩展
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=1, padding=1, groups=expanded_channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels, in_channels, kernel_size=1, bias=False),  # 通道压缩
            nn.BatchNorm2d(in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        semantic_features = self.inverted_bottleneck(x)
        return x + self.sigmoid(semantic_features)  # 残差连接，增强输入特征的语义响应


def finetune_with_kl_divergence(model, train_loader, pretrained_model, alpha=0.5, epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()  # 设置预训练模型为评估模式，不参与梯度更新

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()  # 分类交叉熵损失
    kl_loss = nn.KLDivLoss(reduction="batchmean")  # KL散度损失

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # 微调模型的输出
            outputs = model(data)
            log_probs = F.log_softmax(outputs, dim=1)

            # 预训练模型的输出
            with torch.no_grad():  # 不计算梯度
                pretrained_outputs = pretrained_model(data)
                probs = F.softmax(pretrained_outputs, dim=1)

            # 计算分类损失
            loss_ce = ce_loss(outputs, target)

            # 计算KL散度损失
            loss_kl = kl_loss(log_probs, probs)

            # 总损失 = 分类损失 + alpha * KL散度损失
            loss = loss_ce + alpha * loss_kl

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    print("Training with KL Divergence completed.")
    return model

def split_validate_loader(validate_loader, train_ratio=0.3):
    """
    将验证集切分成每个类别 70% 用于训练，30% 用于验证。
    """
    from collections import defaultdict

    full_dataset = validate_loader.dataset  # 假设 validate_loader 是基于某个 Dataset 对象的

    # 按类别组织数据
    class_data = defaultdict(list)
    for idx in range(len(full_dataset)):
        data, label = full_dataset[idx]
        class_data[label.item()].append((data, label))

    train_data = []
    valid_data = []

    # 对每个类别的数据进行分割
    for label, items in class_data.items():
        total_count = len(items)
        train_count = int(train_ratio * total_count)
        random.shuffle(items)  # 随机打乱
        train_data.extend(items[:train_count])  # 前 70% 用于训练
        valid_data.extend(items[train_count:])  # 剩下的用于验证

    # 自定义数据集类
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    # 创建训练集和验证集 DataLoader
    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)

    train_split_loader = DataLoader(train_dataset, batch_size=validate_loader.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=validate_loader.batch_size, shuffle=False)

    return train_split_loader, valid_loader


class AdaptiveFineTuningModelWithAdapters(nn.Module):
    def __init__(self, base_model, layers_to_finetune, add_sca=True, add_sra=True, device=None):
        super(AdaptiveFineTuningModelWithAdapters, self).__init__()
        self.base_model = base_model
        self.layers_to_finetune = layers_to_finetune  # 需要微调的层

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")



        # 冻结未选择的层
        self._freeze_layers()

    def _freeze_layers(self):
        for name, param in self.base_model.named_parameters():
            if self._is_layer_to_finetune(name):
                param.requires_grad = True  # 解冻选定的微调层
            else:
                param.requires_grad = False  # 冻结未选定的层



    def _is_layer_to_finetune(self, layer_name):
        return any(layer in layer_name for layer in self.layers_to_finetune)

    def forward(self, x):
        x_1 = self.base_model.conv3d_spatial(x)
        x_2 = self.base_model.conv3d_spectral(x)
        x_3 = self.base_model.conv3d_spa_ape(x)

        x = self.base_model.msaa(x_1, x_2, x_3)

        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.base_model.positional_encoding(x)
        x = self.base_model.conv2d_features_1(x)

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = rearrange(x, 'b c h -> b h c')

        # Transformer 及其后的部分（强制解冻参与微调）
        x = self.base_model.cross_attention(x)
        x = x.mean(dim=1)
        x = F.gelu(x)
        x = self.base_model.feed_forward(x)
        x = self.base_model.layer_norm(x)
        x = self.base_model.nn1(x)  # 保证 linear 分类器一定会参与微调
        return x
# 验证模型指标：计算 OA 和 AA
def validate_model_with_score(model, validate_loader, alpha=0.7):
    model.eval()
    total_samples, total_correct = 0, 0
    class_correct, class_total = {}, {}

    device = model.device  # 确保验证时使用模型的设备
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

            for label in torch.unique(target):
                mask = target == label
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                    class_total[label.item()] = 0
                class_correct[label.item()] += (predicted[mask] == target[mask]).sum().item()
                class_total[label.item()] += mask.sum().item()

    oa = total_correct / total_samples
    aa = sum([class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in class_correct]) / len(class_correct)
    score = alpha * aa + (1 - alpha) * oa
    return score

# 粒子群优化
def particle_swarm_optimization(model, train_loader, validate_loader,
                                initial_num_particles=25, initial_max_iterations=10,
                                finetune_epochs=3, inertia_weight=(0.9, 0.4),
                                escape_threshold=5, escape_prob=0.1, aa_threshold=0.01):
    """
    改进版粒子群优化，支持自适应调整粒子数、迭代次数和种群大小。
    """
    # 初始参数
    num_particles = initial_num_particles
    max_iterations = initial_max_iterations

    # 定义参数上限
    MAX_NUM_PARTICLES = 50
    MAX_ITERATIONS = 20

    # 定义早停计数器
    no_improvement_count = 0
    MAX_NO_IMPROVEMENT = 3  # 允许连续多少次AA变化小于阈值后停止调整

    # 初始AA
    previous_aa = 0.0

    # 获取所有层名称
    layer_names = [name for name, _ in model.named_parameters()]
    num_layers = len(layer_names)

    # 初始化粒子的位置和速度
    particles = [random_layer_selection(model) for _ in range(num_particles)]
    velocities = [random.sample(layer_names, k=random.randint(1, len(layer_names) // 2)) for _ in range(num_particles)]

    # 确保 velocities 的长度与 particles 一致
    if len(velocities) != len(particles):
        velocities = [random.sample(layer_names, k=random.randint(1, len(layer_names) // 2)) for _ in range(len(particles))]

    # 个人最优解和全局最优解
    personal_best = particles[:]
    personal_best_scores = [float('-inf')] * num_particles
    global_best = None
    global_best_score = float('-inf')

    # 动态惯性权重
    w_max, w_min = inertia_weight
    w = w_max  # 初始惯性权重

    # 连续未更新全局最优解的计数器
    no_update_count = 0

    for iteration in range(max_iterations):
        for i in range(len(particles)):  # 使用 len(particles) 而不是 num_particles
            # 使用适配器和微调模型
            model_finetuned = AdaptiveFineTuningModelWithAdapters(model, particles[i])
            finetune_model(model_finetuned, train_loader, epochs=finetune_epochs)
            score = validate_model_with_score(model_finetuned, validate_loader)

            # 更新个人最优
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i]

            # 更新全局最优
            if score > global_best_score:
                global_best_score = score
                global_best = particles[i]
                no_update_count = 0  # 重置计数器
            else:
                no_update_count += 1  # 未更新全局最优解

        # 计算当前AA
        current_aa = global_best_score

        # 根据AA的变化调整参数
        if abs(current_aa - previous_aa) < aa_threshold:
            no_improvement_count += 1
            if no_improvement_count >= MAX_NO_IMPROVEMENT:
                print("AA变化连续多次小于阈值，停止调整参数。")
            else:
                # 动态调整参数
                num_particles = min(int(num_particles * 1.3), MAX_NUM_PARTICLES)
                max_iterations = min(int(max_iterations * 1.3), MAX_ITERATIONS)

                # 确保参数至少为 1
                num_particles = max(num_particles, 1)
                max_iterations = max(max_iterations, 1)

                print(
                    f"AA变化较小，调整参数：num_particles={num_particles}, max_iterations={max_iterations}")

                # 动态调整 aa_threshold
                aa_threshold *= 0.9  # 逐渐减小阈值
                print(f"AA变化较小，调整 aa_threshold 为 {aa_threshold}")
        else:
            no_improvement_count = 0  # 重置计数器

        # 更新 previous_aa
        previous_aa = current_aa

        # 防止局部最优：触发粒子扰动
        if no_update_count >= escape_threshold:
            print("Escape triggered: introducing random perturbation to avoid local optimum.")
            for i in range(len(particles)):  # 使用 len(particles) 而不是 num_particles
                if random.random() < escape_prob:
                    particles[i] = random_layer_selection(model)  # 重新初始化部分粒子位置

        # 更新粒子的位置和速度
        for i in range(len(particles)):  # 使用 len(particles) 而不是 num_particles
            # 修改：确保 velocities 和 particles 更新时一致
            velocities[i] = random.sample(layer_names, k=random.randint(1, len(layer_names) // 2))
            new_position = list(set(personal_best[i] + velocities[i]))
            particles[i] = random.sample(new_position, k=min(len(new_position), num_layers))

        # 动态调整惯性权重
        w = w_max - (w_max - w_min) * (iteration / max_iterations)

        print(f"Iteration {iteration}, 当前全局最优分数: {global_best_score}, 惯性权重: {w:.4f}")

    return global_best, global_best_score


# 差分进化
def differential_evolution(model, train_loader, validate_loader, initial_solution, initial_score,
                           max_generations=15, population_size=20, finetune_epochs=3,
                           patience=5, escape_prob=0.1, initial_mutation_rate=0.9, final_mutation_rate=0.4):

    # 初始化参数
    best_solution = initial_solution
    best_score = initial_score
    population = [best_solution] + [random_layer_selection(model) for _ in range(population_size - 1)]
    no_improvement_count = 0
    restart = False  # 用于标志是否需要重新开始差分进化

    def adaptive_mutation_rate(gen):
        return initial_mutation_rate + (final_mutation_rate - initial_mutation_rate) * (gen / max_generations)

    for gen in range(max_generations):
        if restart:
            print(f"Generation {gen}: 由于父代无效，正在重新开始差分进化...")
            # 重新初始化种群并重置计数
            population = [best_solution] + [random_layer_selection(model) for _ in range(population_size - 1)]
            no_improvement_count = 0
            restart = False  # 重置标志位

        new_population = []

        # 精英策略：直接保留当前最优解
        new_population.append(best_solution)

        for i in range(population_size - 1):
            parent1, parent2, parent3 = select_parents(population, i)

            # 如果选择的父代无效，则跳过当前代并重新开始
            if parent1 is None or parent2 is None or parent3 is None:
                restart = True  # 标志位设置为True，表示需要重新开始
                break  # 跳过当前代的交叉变异操作

            # 动态调整变异率
            mutation_rate = adaptive_mutation_rate(gen)

            # 交叉与变异操作
            offspring = crossover_and_mutate(parent1, parent2, parent3, mutation_rate)

            # 微调模型并计算分数
            model_finetuned = AdaptiveFineTuningModelWithAdapters(model, offspring)
            finetune_model(model_finetuned, train_loader, epochs=finetune_epochs)

            score = validate_model_with_score(model_finetuned, validate_loader)

            # 更新最优解
            if score > best_score:
                best_score = score
                best_solution = offspring
                no_improvement_count = 0
                print(f"Generation {gen}, 当前最优解层: {best_solution}, 最优分数: {best_score:.4f}")
            else:
                no_improvement_count += 1

            new_population.append(offspring)

        # 检查早停条件
        if no_improvement_count >= patience:
            print(f"Early stopping triggered at generation {gen}")
            break

        # 检查局部最优并触发逃逸
        if no_improvement_count >= patience // 2:
            print(f"Escape triggered at generation {gen}")
            population = escape_local_optima(population, model, escape_prob)
            no_improvement_count = 0

        population = new_population
        print(f"Generation {gen}, 当前最优分数: {best_score:.4f}")

    return best_solution, best_score



def random_layer_selection(model):
    layer_names = [name for name, _ in model.named_parameters()]
    num_layers = len(layer_names)
    # 随机选择从后往前连续的层数
    end_layer = random.randint(1, num_layers)
    selected_layers = layer_names[-end_layer:]
    return selected_layers

def crossover_and_mutate(parent1, parent2, parent3, mutation_rate=0.9):
    """
    交叉和变异操作。
    """
    if len(parent1) < 2:
        raise ValueError("父代长度太小，不能进行交叉和变异操作。")

    # 交叉点应该在父代的长度内
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]

    if random.random() < mutation_rate:
        mutation_index = random.randint(0, len(offspring) - 1)
        offspring[mutation_index] = random.choice(parent3)

    return offspring

def select_parents(population, index):
    """
    选择三个不同的父代，避免选择无效的父代。
    """
    try:
        parent1 = population[index]
        parent2 = population[(index + 1) % len(population)]
        parent3 = population[(index + 2) % len(population)]

        # 确保父代的长度大于 1
        if len(parent1) < 2 or len(parent2) < 2 or len(parent3) < 2:
            raise ValueError("父代层数不足，不能进行交叉操作。")

        return parent1, parent2, parent3

    except ValueError as e:
        # 捕获异常并打印日志，跳过当前选择
        print(f"Warning: {e} Skipping this parent selection.")
        # 返回 None 表示当前选择无效
        return None, None, None

def escape_local_optima(population, model, escape_prob=0.1):
    """
    局部最优逃逸，通过随机重新初始化部分种群实现。
    """
    for i in range(len(population)):
        if random.random() < escape_prob:
            population[i] = random_layer_selection(model)
    return population

def finetune_model(model, train_loader, epochs=3, lr=1e-6):
    """
    微调模型。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = model.device  # 确保优化时使用模型的设备
    model.train()

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    return model


def optimize_model_with_pso_and_de(model, train_loader, validate_loader, initial_num_particles=10,
                                  initial_max_iterations=5, initial_max_generations=4, initial_population_size=8,
                                  finetune_epochs=4, aa_threshold=0.01):
    print("开始粒子群优化...")
    train_split_loader, valid_loader = split_validate_loader(validate_loader)

    # 初始参数
    num_particles = initial_num_particles
    max_iterations = initial_max_iterations
    population_size = initial_population_size

    # 定义参数上限
    MAX_NUM_PARTICLES = 30
    MAX_ITERATIONS = 20
    MAX_POPULATION_SIZE = 30

    # 定义早停计数器
    no_improvement_count = 0
    MAX_NO_IMPROVEMENT = 3  # 允许连续多少次AA变化小于阈值后停止调整

    # 初始AA
    previous_aa = 0.0

    # PSO阶段
    pso_selected_layers, pso_best_score = particle_swarm_optimization(
        model, train_split_loader, valid_loader,
        initial_num_particles=num_particles,  # 修改为 initial_num_particles
        initial_max_iterations=max_iterations,
        finetune_epochs=finetune_epochs,
        inertia_weight=(0.9, 0.4),  # 动态惯性权重范围，(初始值, 最小值)
        escape_threshold=5,  # 连续未更新全局最优的次数阈值
        escape_prob=0.1  # 扰动的概率，例如 10%
    )
    print(f"PSO阶段选出的候选层: {pso_selected_layers}")
    print(f"PSO阶段最优分数: {pso_best_score}")

    # 计算当前AA
    current_aa = pso_best_score

    # 根据AA的变化调整参数
    if abs(current_aa - previous_aa) < aa_threshold:
        no_improvement_count += 1
        if no_improvement_count >= MAX_NO_IMPROVEMENT:
            print("AA变化连续多次小于阈值，停止调整参数。")
        else:
            # 动态调整参数
            num_particles = min(int(num_particles * 1.5), MAX_NUM_PARTICLES)
            max_iterations = min(int(max_iterations * 1.5), MAX_ITERATIONS)
            population_size = min(int(population_size * 1.5), MAX_POPULATION_SIZE)

            # 确保参数至少为 1
            num_particles = max(num_particles, 1)
            max_iterations = max(max_iterations, 1)
            population_size = max(population_size, 1)

            print(
                f"AA变化较小，调整参数：num_particles={num_particles}, max_iterations={max_iterations}, population_size={population_size}")

            # 动态调整 aa_threshold
            aa_threshold *= 0.9  # 逐渐减小阈值
            print(f"AA变化较小，调整 aa_threshold 为 {aa_threshold}")
    else:
        no_improvement_count = 0  # 重置计数器

    # 更新 previous_aa
    previous_aa = current_aa

    print("开始差分进化优化...")
    de_best_layers, de_best_score = differential_evolution(
        model, train_split_loader, valid_loader,
        initial_solution=pso_selected_layers,
        initial_score=pso_best_score,
        max_generations=initial_max_generations,
        population_size=population_size,
        finetune_epochs=finetune_epochs
    )

    print(f"差分进化阶段选出的候选层: {de_best_layers}")
    print(f"差分进化阶段最优分数: {de_best_score}")

    return de_best_layers, de_best_score

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


    train_samples_per_class = {}  
    validate_samples_per_class = {}  

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# L2正则化函数
def l2_regularization(model, lambda_l2=1e-4):
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return lambda_l2 * l2_norm


# 训练函数（加入KL散度）
# 训练函数（加入KL散度）
def train_with_kl(model, train_loader, validate_loader, pretrained_model, epochs, alpha=0.3, lr=0.001, lambda_l2=1e-4):
    """
    使用KL散度与交叉熵联合损失进行模型训练，支持动态学习率调整。

    参数：
        model: 待训练模型。
        train_loader: 训练集加载器。
        validate_loader: 验证集加载器。
        pretrained_model: 冻结的预训练模型，用于提供概率分布。
        epochs: 训练轮数。
        alpha: KL散度的权重系数。
        lr: 初始学习率。
        lambda_l2: L2正则化系数。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移动到设备
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()  # 冻结预训练模型

    # 定义损失函数
    ce_loss = nn.CrossEntropyLoss()  # 分类交叉熵损失
    kl_loss = nn.KLDivLoss(reduction="batchmean")  # KL散度损失
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)

    # 添加学习率调度器，根据验证集损失动态调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    current_loss_high_count = 0  # 损失计数器

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 模型的预测输出
            outputs = model(data)
            log_probs = F.log_softmax(outputs, dim=1)

            # 预训练模型的预测分布
            with torch.no_grad():
                pretrained_outputs = pretrained_model(data)
                probs = F.softmax(pretrained_outputs, dim=1)

            # 计算分类损失
            loss_ce = ce_loss(outputs, target)

            # 计算KL散度损失
            loss_kl = kl_loss(log_probs, probs)

            # 总损失 = 分类损失 + alpha * KL散度损失 + L2正则化
            loss = loss_ce + alpha * loss_kl + l2_regularization(model, lambda_l2)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (
            epoch + 1,
            total_loss / (i + 1),
            loss.item()
        ))

        # 验证模型并获取验证损失
        validate_loss = validate(validate_loader, model, ce_loss, device)

        # 根据验证损失调整学习率
        scheduler.step(validate_loss)

        # 检查当前损失
        if loss.item() > 1.5:
            current_loss_high_count += 1
            print(f"Current loss > 1.5: Count = {current_loss_high_count}")
        else:
            current_loss_high_count = 0  # 重置计数器

        # 如果连续10个epoch损失高于1.5，则重新开始训练
        if current_loss_high_count >= 50:
            print("Current loss has been above 3 for 10 epochs, restarting training.")
            return train_with_kl(model, train_loader, validate_loader, pretrained_model, epochs, alpha, lr, lambda_l2)

        # 每20轮保存模型
        if (epoch + 1) % 20 == 0:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d')
            model_path = f'_{current_time}.pth'

            # 清理权重名称中的前缀 'base_model.'
            state_dict = model.state_dict()
            cleaned_state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}
            torch.save(cleaned_state_dict, model_path)  # 保存清理后的权重
            print(f"Model saved at epoch {epoch + 1}, time: {current_time}")

    print('Finished Training with KL Divergence')
    return model, device

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
    pretrained_model_path = 'path/your_pretrained_model.pth'
    tic1 = time.perf_counter()
    # 加载数据
    train_loader, validate_loader, test_loader, all_data_loader, y_all, gt = create_data_loader()

    # 加载预训练模型
    base_model = DSP2Net.DSP2Net().cuda()
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    base_model.load_state_dict(checkpoint)
    print(f"Pretrained model loaded from {pretrained_model_path}")

    # PSO 和 DE 联合优化
    best_layers, best_score = optimize_model_with_pso_and_de(base_model, train_loader, validate_loader)

    # 使用优化结果微调模型
    finetuned_model = AdaptiveFineTuningModelWithAdapters(
        base_model,
        best_layers,
        add_sca=True,
        add_sra=True
    ).cuda()

    # 训练微调后的模型
    net, device = train_with_kl(finetuned_model, train_loader, validate_loader, pretrained_model=base_model,  # 冻结的预训练模型
    epochs=60,
    alpha=0.5,
    lr=0.0001,
    lambda_l2=1e-4
)

    toc1 = time.perf_counter()  # 结束训练时间记录

    # 记录测试时间
    tic2 = time.perf_counter()

    # 测试模型
    y_pred_test, y_test = test(device, net, all_data_loader, save_pred_path='predictions.mat',
                               save_label_path='labels.mat')

    toc2 = time.perf_counter()  # 结束测试时间记录

    # 评估性能
    classification, oa, confusion, each_acc, aa, kappa, recall, f1, IOU, FWIOU = acc_reports(y_test, y_pred_test)
    classification = str(classification)

    # 计算训练和测试的时间
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2

    # 定义标签和调色板
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
        1: (55, 166, 2),  # 类别1的颜色 (绿色)
        2: (255, 255, 195),  # 类别2的颜色 (黄色)
        3: (250, 250, 127),  # 类别3的颜色 (蓝色)
        4: (168, 168, 0),  # 类别4的颜色 (橙色)
        5: (6, 110, 255),  # 类别5的颜色 (紫色)
        6: (251, 4, 0),  # 类别6的颜色 (红色)
        7: (230, 230, 230),  # 类别7的颜色 (青色)
        8: (178, 178, 178),  # 类别8的颜色 (灰色)
        9: (129, 129, 129),  # 类别9的颜色 (深灰色)
    }

    # 可选：打印调色板以验证分配的颜色
    for k, color in palette.items():
        print(f"Label {k}: Color {color}")

    # 使用测试集进行测试
    y_pred_test, y_new = test(device, net, all_data_loader)
    print(f"Predicted labels shape: {y_pred_test.shape}")

    # 获取分类结果
    cls_labels = get_classification_map(y_pred_test, gt)

    # 将分类标签转为颜色标签
    color_cls_labels = convert_to_color(cls_labels)

    # 创建保存结果的文件夹（如果不存在）
    folder_path = "cls_map"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存分类结果图像
    MODEL = "model"  # 模型名称
    DATASET = "1"  # 数据集名称
    run = 1  # 运行编号
    rounded_value = round(oa, 2)
    rounded_value_2 = round(aa, 2)

    # 图片名称
    im_name = f"{MODEL}_{DATASET}_{run}_{rounded_value}_{rounded_value_2}.jpg"
    im = Image.fromarray(color_cls_labels)
    im = im.resize((im.width, im.height), Image.NEAREST)
    im.save(os.path.join(folder_path, im_name))

    # 打印训练和测试时间
    print('Training Time (s):', Training_Time)
    print('Test Time (s):', Test_time)

    # 打印各项性能指标
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
