"""
NAS-Bench-201 NTK Condition Number & K-Score 相关性实验

随机采样500个架构，计算NTK条件数和K-score，
通过API查询12轮和200轮CIFAR-10精度，
保存CSV并绘制相关系数图。
"""

import os
import sys
import random
import time
import csv
import gc

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from nats_bench import create

# ============================================================
# NAS-Bench-201 Network Implementation
# (替代 xautodl，直接实现 NAS-Bench-201 的 cell-based tiny net)
# ============================================================

# NAS-Bench-201 的 5 种操作
NAS201_OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class NAS201Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class NAS201Identity(nn.Module):
    def forward(self, x):
        return x


class NAS201ResNetBasicBlock(nn.Module):
    """NAS-Bench-201 style residual block used in InferCell for reduction."""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        assert stride in [1, 2]
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride, 1)
        self.conv_b = ReLUConvBN(C_out, C_out, 3, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            )
        elif C_in != C_out:
            self.downsample = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock


def get_nas201_op(op_name, C_in, C_out, stride):
    """根据操作名称返回对应的 nn.Module"""
    if op_name == 'none':
        return NAS201Zero(stride)
    elif op_name == 'skip_connect':
        if C_in == C_out and stride == 1:
            return NAS201Identity()
        else:
            return NAS201ResNetBasicBlock(C_in, C_out, stride)
    elif op_name == 'nor_conv_1x1':
        return ReLUConvBN(C_in, C_out, 1, stride, 0)
    elif op_name == 'nor_conv_3x3':
        return ReLUConvBN(C_in, C_out, 3, stride, 1)
    elif op_name == 'avg_pool_3x3':
        if C_in == C_out and stride == 1:
            return nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        else:
            return nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                nn.Conv2d(C_in, C_out, 1, bias=False),
            )
    else:
        raise ValueError(f'Unknown op: {op_name}')


class InferCell(nn.Module):
    """
    NAS-Bench-201 的推理 Cell。
    4 个节点 (node0=input, node1, node2, node3)，6 条边。
    每个节点的输出 = sum(对应边的操作输出)
    """
    def __init__(self, op_names, C_in, C_out, stride):
        super().__init__()
        # op_names: 6个操作名称，按边的顺序
        # edges: (0->1), (0->2), (1->2), (0->3), (1->3), (2->3)
        self.op_0_1 = get_nas201_op(op_names[0], C_in, C_out, stride)
        self.op_0_2 = get_nas201_op(op_names[1], C_in, C_out, stride)
        self.op_1_2 = get_nas201_op(op_names[2], C_out, C_out, 1)
        self.op_0_3 = get_nas201_op(op_names[3], C_in, C_out, stride)
        self.op_1_3 = get_nas201_op(op_names[4], C_out, C_out, 1)
        self.op_2_3 = get_nas201_op(op_names[5], C_out, C_out, 1)

    def forward(self, x):
        node0 = x
        node1 = self.op_0_1(node0)
        node2 = self.op_0_2(node0) + self.op_1_2(node1)
        node3 = self.op_0_3(node0) + self.op_1_3(node1) + self.op_2_3(node2)
        return node3


class NAS201TinyNet(nn.Module):
    """
    NAS-Bench-201 的完整网络。
    结构: stem -> N cells (stage1) -> resblock -> N cells (stage2) -> resblock -> N cells (stage3) -> gap -> fc
    默认: N=5, channels=[16, 32, 64], num_classes=10
    """
    def __init__(self, op_names, num_classes=10, N=5, channels=(16, 32, 64)):
        super().__init__()
        self.channels = channels
        self.N = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )

        # Build cells
        layers = []
        C_in = channels[0]
        for stage_idx, C_out in enumerate(channels):
            if stage_idx > 0:
                # Reduction: residual block with stride 2
                layers.append(NAS201ResNetBasicBlock(C_in, C_out, stride=2))
                C_in = C_out
            for _ in range(N):
                layers.append(InferCell(op_names, C_in, C_out, stride=1))
                C_in = C_out

        self.cells = nn.Sequential(*layers)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_in), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_in, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = self.lastact(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def parse_arch_str(arch_str):
    """
    解析 NAS-Bench-201 架构字符串，返回 6 个操作名称。
    格式: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    """
    # Split by '+' to get node groups
    nodes = arch_str.split('+')
    op_names = []
    for node_str in nodes:
        # Extract op names from |op~source| patterns
        parts = node_str.strip().split('|')
        for part in parts:
            part = part.strip()
            if '~' in part:
                op_name = part.split('~')[0]
                op_names.append(op_name)
    return op_names


# ============================================================
# NTK & K-Score Computation (复用 src/engine/evaluator.py 逻辑)
# ============================================================

def get_cifar10_loader(data_root='./data', batch_size=64):
    """获取 CIFAR-10 数据加载器（用于 NTK/K 计算）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader


def compute_ntk_cond(network, loader, device='cuda', num_batch=1):
    """
    计算 NTK 条件数。
    复用 src/engine/evaluator.py 中 NTKEvaluator._compute_ntk_eigenvalues 的逻辑。
    """
    network = network.to(device)
    network.train()

    grads = []
    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batch:
            break
        inputs = inputs.to(device, non_blocking=True)
        network.zero_grad()
        inputs_ = inputs.clone().to(device, non_blocking=True)
        logit = network(inputs_)
        if isinstance(logit, tuple):
            logit = logit[1]

        for _idx in range(len(inputs_)):
            logit[_idx:_idx + 1].backward(torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            grads.append(torch.cat(grad, -1))
            network.zero_grad()
            torch.cuda.empty_cache()

    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', [grads, grads])

    try:
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U')
        eigenvalues = torch.clamp(eigenvalues, min=1e-30)

        if torch.isnan(eigenvalues).any():
            return None

        if eigenvalues[-1].item() < 1e-8:
            return None

        cond = (eigenvalues[-1] / eigenvalues[0]).item()
        if np.isnan(cond) or np.isinf(cond):
            return None
        return cond
    except Exception as e:
        print(f'NTK eigenvalue computation failed: {e}')
        return None


def compute_k_score(network, loader, device='cuda', num_batch=1, eps=1e-6):
    """
    计算 K-score (ReLU kernel logdet)。
    复用 src/engine/evaluator.py 中 NTKEvaluator._compute_k_logdet 的逻辑。
    """
    network = network.to(device)
    network.eval()

    handles = []
    k_holder = {'K': None}

    def counting_hook(module, inp, out):
        x = (out > 0).float()
        x = x.view(x.size(0), -1)
        k1 = x @ x.t()
        k2 = (1.0 - x) @ (1.0 - x).t()
        if k_holder['K'] is not None:
            k_holder['K'] += k1 + k2
        else:
            k_holder['K'] = k1 + k2

    for _, module in network.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(counting_hook))

    try:
        with torch.no_grad():
            for i, (inputs, _) in enumerate(loader):
                if i >= num_batch:
                    break
                inputs = inputs.to(device, non_blocking=True)
                _ = network(inputs)

        if k_holder['K'] is None:
            return None

        k = k_holder['K']
        k = k + eps * torch.eye(k.size(0), device=k.device, dtype=k.dtype)
        sign, logdet = torch.linalg.slogdet(k.double())
        if sign <= 0:
            return None
        return logdet.item()
    except Exception as e:
        print(f'K computation failed: {e}')
        return None
    finally:
        for h in handles:
            h.remove()


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# Main Experiment
# ============================================================

def main():
    # ---- 配置 ----
    SAMPLE_SIZE = 50
    SEED = 42
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    BATCH_SIZE = 64
    NUM_BATCH = 1  # NTK/K 计算用的 batch 数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # NAS-Bench-201 API 数据文件路径
    # 用户需要将 NATS-tss-v1_0-3ffb9-simple 文件放在此位置
    NATS_BENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NATS-tss-v1_0-3ffb9-simple', 'NATS-tss-v1_0-3ffb9-simple')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"NATS-Bench path: {NATS_BENCH_PATH}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")

    # ---- 加载 NAS-Bench-201 API ----
    print("\n[1/5] Loading NAS-Bench-201 API...")
    api = create(NATS_BENCH_PATH, 'tss', fast_mode=True, verbose=False)
    total_archs = len(api)
    print(f"Total architectures in search space: {total_archs}")

    # ---- 随机采样 ----
    print(f"\n[2/5] Sampling {SAMPLE_SIZE} architectures...")
    sampled_indices = random.sample(range(total_archs), min(SAMPLE_SIZE, total_archs))
    print(f"Sampled {len(sampled_indices)} architectures.")

    # ---- 加载 CIFAR-10 数据 ----
    print("\n[3/5] Loading CIFAR-10 data...")
    loader = get_cifar10_loader(data_root=DATA_ROOT, batch_size=BATCH_SIZE)
    print("CIFAR-10 loaded.")

    # ---- 计算 NTK / K-score / 查询精度 ----
    print(f"\n[4/5] Computing NTK condition number, K-score, and querying accuracy...")
    results = []
    fail_count = 0

    for count, idx in enumerate(sampled_indices):
        arch_str = api.arch(idx)
        op_names = parse_arch_str(arch_str)

        # 查询精度
        try:
            acc_12, _, _, _ = api.simulate_train_eval(idx, dataset='cifar10', hp='12')
            acc_200, _, _, _ = api.simulate_train_eval(idx, dataset='cifar10', hp='200')
        except Exception as e:
            print(f"  [{count+1}/{SAMPLE_SIZE}] API query failed for idx={idx}: {e}")
            fail_count += 1
            continue

        # 构建网络
        try:
            network = NAS201TinyNet(op_names, num_classes=10, N=5, channels=(16, 32, 64))
        except Exception as e:
            print(f"  [{count+1}/{SAMPLE_SIZE}] Network build failed for idx={idx}: {e}")
            fail_count += 1
            continue

        # 计算 NTK 条件数
        try:
            ntk_cond = compute_ntk_cond(network, loader, device=DEVICE, num_batch=NUM_BATCH)
        except Exception as e:
            print(f"  [{count+1}/{SAMPLE_SIZE}] NTK computation failed: {e}")
            ntk_cond = None

        # 计算 K-score
        try:
            k_score = compute_k_score(network, loader, device=DEVICE, num_batch=NUM_BATCH)
        except Exception as e:
            print(f"  [{count+1}/{SAMPLE_SIZE}] K computation failed: {e}")
            k_score = None

        del network
        clear_gpu_memory()

        results.append({
            'arch_index': idx,
            'arch_str': arch_str,
            'ntk_cond': ntk_cond,
            'k_score': k_score,
            'acc_12epoch': acc_12,
            'acc_200epoch': acc_200,
        })

        if (count + 1) % 10 == 0 or count == 0:
            ntk_str = f"{ntk_cond:.4f}" if ntk_cond is not None else "FAIL"
            k_str = f"{k_score:.4f}" if k_score is not None else "FAIL"
            print(f"  [{count+1}/{SAMPLE_SIZE}] idx={idx} | NTK={ntk_str} | "
                  f"K={k_str} | Acc@12={acc_12:.2f} | Acc@200={acc_200:.2f}")

    print(f"\nDone. {len(results)} successful, {fail_count} failed.")

    # ---- 保存 CSV ----
    csv_path = os.path.join(RESULTS_DIR, 'nas201_ntk_kscore_data.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['arch_index', 'arch_str', 'ntk_cond', 'k_score',
                                               'acc_12epoch', 'acc_200epoch'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to: {csv_path}")

    # ---- 绘制相关系数图 ----
    print("\n[5/5] Plotting correlation figures...")
    plot_correlations(results, RESULTS_DIR)
    print("All done!")


def plot_correlations(results, save_dir):
    """绘制 NTK/K-score 与精度的相关系数散点图"""
    # 过滤有效数据
    valid = [r for r in results if r['ntk_cond'] is not None and r['k_score'] is not None]
    if len(valid) < 3:
        print("WARNING: Not enough valid data to plot correlations.")
        return

    ntk_vals = np.array([r['ntk_cond'] for r in valid])
    k_vals = np.array([r['k_score'] for r in valid])
    acc_12 = np.array([r['acc_12epoch'] for r in valid])
    acc_200 = np.array([r['acc_200epoch'] for r in valid])

    # 对NTK取log以便更好地可视化
    ntk_log = np.log10(ntk_vals + 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('NAS-Bench-201: NTK Condition Number & K-Score vs. Accuracy\n'
                 f'({len(valid)} architectures on CIFAR-10)',
                 fontsize=16, fontweight='bold')

    pairs = [
        (ntk_log, acc_12, 'log₁₀(NTK Condition Number)', 'Accuracy @ 12 epochs (%)', axes[0, 0]),
        (ntk_log, acc_200, 'log₁₀(NTK Condition Number)', 'Accuracy @ 200 epochs (%)', axes[0, 1]),
        (k_vals, acc_12, 'K-Score (log det)', 'Accuracy @ 12 epochs (%)', axes[1, 0]),
        (k_vals, acc_200, 'K-Score (log det)', 'Accuracy @ 200 epochs (%)', axes[1, 1]),
    ]

    for x_data, y_data, xlabel, ylabel, ax in pairs:
        # 散点图
        scatter = ax.scatter(x_data, y_data, alpha=0.35, s=15, c='#2196F3', edgecolors='none')

        # 计算相关系数
        spearman_r, spearman_p = stats.spearmanr(x_data, y_data)
        kendall_t, kendall_p = stats.kendalltau(x_data, y_data)
        pearson_r, pearson_p = stats.pearsonr(x_data, y_data)

        # 拟合趋势线
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2, label='Linear fit')

        # 标注相关系数
        text = (f'Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})\n'
                f'Kendall  τ = {kendall_t:.4f} (p={kendall_p:.2e})\n'
                f'Pearson  r = {pearson_r:.4f} (p={pearson_p:.2e})')
        ax.text(0.03, 0.97, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(save_dir, 'nas201_correlation.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Correlation plot saved to: {plot_path}")

    # ---- 额外：相关系数汇总表 ----
    print("\n" + "=" * 60)
    print("Correlation Coefficients Summary")
    print("=" * 60)
    print(f"{'Metric Pair':<40} {'Spearman ρ':>12} {'Kendall τ':>12}")
    print("-" * 60)

    for x_data, y_data, xlabel, ylabel, _ in pairs:
        sp_r, _ = stats.spearmanr(x_data, y_data)
        kt_t, _ = stats.kendalltau(x_data, y_data)
        name = f"{xlabel.split('(')[0].strip()} vs {ylabel}"
        print(f"{name:<40} {sp_r:>12.4f} {kt_t:>12.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
