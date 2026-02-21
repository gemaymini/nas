"""
NTK条件数与K score互补性实验

实验目的：验证NTK条件数和K score在架构评估中的互补关系。
- 纯无参数操作架构：NTK条件数低（好），但K score低（差）
- 纯有参数操作架构：K score高（好），但NTK条件数高（差）  
- 均衡架构：两指标均处于合理范围

实验方法：构造三类极端架构并随机采样，计算对比两个指标。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from configuration.config import config
from core.encoding import Individual, CellEncoding, Edge
from models.network import Network
from engine.evaluator import NTKEvaluator, clear_gpu_memory

# 操作分类
# 无参数操作（Parameter-free）: zero(0), skip_connect(1), max_pool_3x3(6), avg_pool_3x3(7)
# 有参数操作（Parameterized）: sep_conv_3x3(2), sep_conv_5x5(3), dil_conv_3x3(4), dil_conv_5x5(5)
PARAM_FREE_OPS = [1, 6, 7]      # skip_connect, max_pool, avg_pool (排除zero以免死网络)
PARAMETERIZED_OPS = [2, 3, 4, 5] # sep_conv_3x3, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5
ALL_VALID_OPS = PARAM_FREE_OPS + PARAMETERIZED_OPS


def make_cell(op_pool):
    """用指定的操作集合随机构建一个cell"""
    edges = []
    for node_idx in range(config.NUM_NODES):
        node_edges = []
        for _ in range(config.EDGES_PER_NODE):
            valid_sources = list(range(2 + node_idx))
            source = random.choice(valid_sources)
            op_id = random.choice(op_pool)
            node_edges.append(Edge(source=source, op_id=op_id))
        edges.append(node_edges)
    return CellEncoding(edges=edges)


def make_individual(op_pool):
    """用指定的操作集合随机构建一个Individual"""
    normal_cell = make_cell(op_pool)
    reduction_cell = make_cell(op_pool)
    return Individual(normal_cell=normal_cell, reduction_cell=reduction_cell)


def make_mixed_individual(param_free_ratio=0.5):
    """构建混合架构，param_free_ratio为无参数操作的比例"""
    def make_mixed_cell():
        edges = []
        for node_idx in range(config.NUM_NODES):
            node_edges = []
            for _ in range(config.EDGES_PER_NODE):
                valid_sources = list(range(2 + node_idx))
                source = random.choice(valid_sources)
                if random.random() < param_free_ratio:
                    op_id = random.choice(PARAM_FREE_OPS)
                else:
                    op_id = random.choice(PARAMETERIZED_OPS)
                node_edges.append(Edge(source=source, op_id=op_id))
            edges.append(node_edges)
        return CellEncoding(edges=edges)
    
    normal_cell = make_mixed_cell()
    reduction_cell = make_mixed_cell()
    return Individual(normal_cell=normal_cell, reduction_cell=reduction_cell)


def count_ops(individual):
    """统计一个个体中各类操作的比例"""
    param_free_count = 0
    parameterized_count = 0
    total = 0
    for cell in [individual.normal_cell, individual.reduction_cell]:
        for node_edges in cell.edges:
            for edge in node_edges:
                total += 1
                if edge.op_id in PARAM_FREE_OPS or edge.op_id == 0:
                    param_free_count += 1
                else:
                    parameterized_count += 1
    return {
        'param_free_ratio': param_free_count / total if total > 0 else 0,
        'parameterized_ratio': parameterized_count / total if total > 0 else 0,
        'param_free_count': param_free_count,
        'parameterized_count': parameterized_count,
        'total': total
    }


def evaluate_individual(evaluator, individual):
    """评估一个个体的NTK和K score"""
    try:
        network = Network(individual.normal_cell, individual.reduction_cell)
        param_count = network.get_param_count()
        
        ntk_score = evaluator.compute_ntk_score(network)
        k_score = evaluator.compute_k_score(network)
        
        del network
        clear_gpu_memory()
        
        return {
            'ntk_score': ntk_score,
            'k_score': k_score,
            'param_count': param_count
        }
    except Exception as e:
        print(f"  Error evaluating individual {individual.id}: {e}")
        clear_gpu_memory()
        return None


def run_experiment(num_samples_per_category=50):
    """运行互补性实验"""
    print("=" * 70)
    print("NTK条件数与K score互补性实验")
    print("=" * 70)
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    evaluator = NTKEvaluator()
    
    categories = {
        'param_free': {
            'name': '纯无参数操作架构',
            'name_en': 'Parameter-free',
            'generator': lambda: make_individual(PARAM_FREE_OPS),
            'results': []
        },
        'parameterized': {
            'name': '纯有参数操作架构', 
            'name_en': 'Parameterized',
            'generator': lambda: make_individual(PARAMETERIZED_OPS),
            'results': []
        },
        'balanced': {
            'name': '均衡混合架构',
            'name_en': 'Balanced',
            'generator': lambda: make_mixed_individual(0.5),
            'results': []
        }
    }
    
    all_results = {}
    
    for cat_key, cat_info in categories.items():
        print(f"\n{'='*50}")
        print(f"类别: {cat_info['name']} ({cat_info['name_en']})")
        print(f"{'='*50}")
        
        results = []
        for i in range(num_samples_per_category):
            individual = cat_info['generator']()
            ops_info = count_ops(individual)
            
            print(f"  [{i+1}/{num_samples_per_category}] ID={individual.id} "
                  f"(无参数: {ops_info['param_free_ratio']:.1%}, "
                  f"有参数: {ops_info['parameterized_ratio']:.1%})")
            
            result = evaluate_individual(evaluator, individual)
            if result is not None:
                result['ops_info'] = ops_info
                result['individual_id'] = individual.id
                results.append(result)
                print(f"    NTK={result['ntk_score']:.4f}, "
                      f"K={result['k_score']:.4f}, "
                      f"Params={result['param_count']:,}")
            else:
                print(f"    评估失败，跳过")
        
        cat_info['results'] = results
        all_results[cat_key] = results
    
    # 输出统计
    print("\n\n" + "=" * 70)
    print("实验结果统计")
    print("=" * 70)
    print(f"{'类别':<20} {'样本数':>6} {'NTK均值':>12} {'NTK中位数':>12} "
          f"{'K均值':>12} {'K中位数':>12} {'参数量均值':>14}")
    print("-" * 100)
    
    stats = {}
    for cat_key, cat_info in categories.items():
        results = cat_info['results']
        if not results:
            continue
        ntk_scores = [r['ntk_score'] for r in results if r['ntk_score'] < config.NTK_FAIL_SCORE]
        k_scores = [r['k_score'] for r in results if r['k_score'] > config.K_FAIL_SCORE]
        params = [r['param_count'] for r in results]
        
        stat = {
            'n': len(results),
            'ntk_mean': np.mean(ntk_scores) if ntk_scores else float('nan'),
            'ntk_median': np.median(ntk_scores) if ntk_scores else float('nan'),
            'ntk_std': np.std(ntk_scores) if ntk_scores else float('nan'),
            'k_mean': np.mean(k_scores) if k_scores else float('nan'),
            'k_median': np.median(k_scores) if k_scores else float('nan'),
            'k_std': np.std(k_scores) if k_scores else float('nan'),
            'param_mean': np.mean(params),
            'ntk_valid': len(ntk_scores),
            'k_valid': len(k_scores),
        }
        stats[cat_key] = stat
        
        print(f"{cat_info['name']:<20} {stat['n']:>6} {stat['ntk_mean']:>12.2f} "
              f"{stat['ntk_median']:>12.2f} {stat['k_mean']:>12.2f} "
              f"{stat['k_median']:>12.2f} {stat['param_mean']:>14,.0f}")
    
    # 保存结果
    save_dir = os.path.join(config.LOG_DIR, 'complementarity_experiment')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始数据
    save_data = {}
    for cat_key in all_results:
        save_data[cat_key] = []
        for r in all_results[cat_key]:
            save_data[cat_key].append({
                'individual_id': r['individual_id'],
                'ntk_score': r['ntk_score'],
                'k_score': r['k_score'],
                'param_count': r['param_count'],
                'param_free_ratio': r['ops_info']['param_free_ratio'],
                'parameterized_ratio': r['ops_info']['parameterized_ratio'],
            })
    
    with open(os.path.join(save_dir, 'raw_results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 绘图
    plot_results(categories, stats, save_dir)
    
    print(f"\n结果已保存到: {save_dir}")
    return categories, stats


def plot_results(categories, stats, save_dir):
    """绘制实验结果图"""
    
    # 图1: 三类架构的NTK vs K score散点图
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = {'param_free': '#e74c3c', 'parameterized': '#2ecc71', 'balanced': '#3498db'}
    markers = {'param_free': 'o', 'parameterized': 's', 'balanced': '^'}
    labels = {
        'param_free': '纯无参数操作 (skip, pool)',
        'parameterized': '纯有参数操作 (conv)',
        'balanced': '均衡混合架构'
    }
    
    for cat_key, cat_info in categories.items():
        results = cat_info['results']
        if not results:
            continue
        ntk = [r['ntk_score'] for r in results]
        k = [r['k_score'] for r in results]
        ax.scatter(ntk, k, c=colors[cat_key], marker=markers[cat_key],
                  s=80, alpha=0.7, edgecolors='white', linewidth=0.5,
                  label=labels[cat_key])
    
    ax.set_xlabel('NTK条件数 (越低越好)', fontsize=13)
    ax.set_ylabel('K score / log-det (越高越好)', fontsize=13)
    ax.set_title('NTK条件数与K score的互补性分析\n(DARTS搜索空间, CIFAR-10)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加注释区域
    # 左上：理想区域
    # 右下：最差区域
    ax.annotate('理想区域\n(低NTK + 高K)', xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=10, color='green', fontweight='bold',
               ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ntk_vs_kscore_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 图2: 箱线图对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cat_names = []
    ntk_data = []
    k_data = []
    for cat_key in ['param_free', 'balanced', 'parameterized']:
        cat_info = categories[cat_key]
        results = cat_info['results']
        if not results:
            continue
        cat_names.append(labels[cat_key])
        ntk_data.append([r['ntk_score'] for r in results if r['ntk_score'] < config.NTK_FAIL_SCORE])
        k_data.append([r['k_score'] for r in results if r['k_score'] > config.K_FAIL_SCORE])
    
    # NTK箱线图
    bp1 = axes[0].boxplot(ntk_data, labels=['无参数', '均衡', '有参数'],
                          patch_artist=True, widths=0.5)
    box_colors = ['#e74c3c', '#3498db', '#2ecc71']
    for patch, color in zip(bp1['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel('NTK条件数', fontsize=12)
    axes[0].set_title('各类架构的NTK条件数分布\n(越低 → 可训练性越强)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # K score箱线图
    bp2 = axes[1].boxplot(k_data, labels=['无参数', '均衡', '有参数'],
                          patch_artist=True, widths=0.5)
    for patch, color in zip(bp2['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel('K score (log-det)', fontsize=12)
    axes[1].set_title('各类架构的K score分布\n(越高 → 表达能力越强)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('NTK条件数与K score在不同操作偏好架构上的互补特性', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplot_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 图3: 柱状图统计均值对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_labels = ['无参数操作', '均衡混合', '有参数操作']
    x = np.arange(len(x_labels))
    width = 0.4
    
    ntk_means = [stats.get('param_free', {}).get('ntk_mean', 0),
                 stats.get('balanced', {}).get('ntk_mean', 0),
                 stats.get('parameterized', {}).get('ntk_mean', 0)]
    ntk_stds = [stats.get('param_free', {}).get('ntk_std', 0),
                stats.get('balanced', {}).get('ntk_std', 0),
                stats.get('parameterized', {}).get('ntk_std', 0)]
    
    k_means = [stats.get('param_free', {}).get('k_mean', 0),
               stats.get('balanced', {}).get('k_mean', 0),
               stats.get('parameterized', {}).get('k_mean', 0)]
    k_stds = [stats.get('param_free', {}).get('k_std', 0),
              stats.get('balanced', {}).get('k_std', 0),
              stats.get('parameterized', {}).get('k_std', 0)]
    
    bars1 = axes[0].bar(x, ntk_means, width, yerr=ntk_stds, capsize=5,
                        color=box_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, fontsize=11)
    axes[0].set_ylabel('NTK条件数 (均值±标准差)', fontsize=11)
    axes[0].set_title('NTK条件数对比', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    bars2 = axes[1].bar(x, k_means, width, yerr=k_stds, capsize=5,
                        color=box_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, fontsize=11)
    axes[1].set_ylabel('K score (均值±标准差)', fontsize=11)
    axes[1].set_title('K score对比', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bar_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存到 {save_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NTK与K score互补性实验')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='每类架构的采样数量')
    args = parser.parse_args()
    
    categories, stats = run_experiment(num_samples_per_category=args.num_samples)
