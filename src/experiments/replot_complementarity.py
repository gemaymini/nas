"""
重新可视化互补性实验结果
- 修复中文乱码问题
- 不过滤任何NTK值，展示全部数据的互补性
- 使用对数坐标处理NTK的巨大范围差异
- Nature风格配色，学术审美
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
# 中文字体配置
# ============================================================
def setup_chinese_font():
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong',
        'KaiTi', 'STSong', 'STHeiti',
    ]
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            print(f"使用中文字体: {font_name}")
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font_name

    font_dirs = [
        r'C:\Windows\Fonts',
        os.path.expanduser(r'~\AppData\Local\Microsoft\Windows\Fonts'),
    ]
    chinese_font_files = ['simhei.ttf', 'msyh.ttc', 'simsun.ttc', 'simfang.ttf']
    for font_dir in font_dirs:
        for font_file in chinese_font_files:
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return font_name
    print("警告: 未找到中文字体")
    return None

setup_chinese_font()

# ============================================================
# 加载数据
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs', 'complementarity_experiment')
SAVE_DIR = DATA_DIR

with open(os.path.join(DATA_DIR, 'raw_results.json'), 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# ============================================================
# Nature 风格柔和渐变配色
# ============================================================
colors = {
    'param_free':    '#4393C3',   # 柔和蓝
    'parameterized': '#D6604D',   # 柔和红
    'balanced':      '#92C5DE',   # 浅蓝
}
markers = {'param_free': 'o', 'parameterized': 's', 'balanced': '^'}
labels_cn = {
    'param_free': '纯无参数操作 (skip, pool)',
    'parameterized': '纯有参数操作 (conv)',
    'balanced': '均衡混合架构'
}
short_labels = {
    'param_free': '无参数',
    'parameterized': '有参数',
    'balanced': '均衡'
}
order = ['param_free', 'balanced', 'parameterized']
box_colors = [colors[k] for k in order]

# K score 基准偏移量，用于放大纵轴差异
K_OFFSET = 1770

for cat_key, results in raw_data.items():
    ntk = [r['ntk_score'] for r in results]
    ks = [r['k_score'] for r in results]
    print(f"{cat_key}: {len(results)} samples, NTK median={np.median(ntk):.2f}, K median={np.median(ks):.2f}")

# ============================================================
# 图1: NTK vs K score 散点图
#   - 对数x轴 + 标注 log₁₀
#   - 纵轴使用 K_score - 1770 放大差异
#   - 图例放右上角
# ============================================================
print("\n绘制散点图...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for cat_key in ['param_free', 'balanced', 'parameterized']:
    results = raw_data[cat_key]
    ntk = [r['ntk_score'] for r in results]
    k = [r['k_score'] for r in results]
    ax.scatter(ntk, k, c=colors[cat_key], marker=markers[cat_key],
              s=60, alpha=0.55, edgecolors='white', linewidth=0.4,
              label=f"{labels_cn[cat_key]} (n={len(results)})", zorder=3)

ax.set_xscale('log')
ax.set_ylim(bottom=K_OFFSET)

ax.set_xlabel(r'NTK条件数 ($\log_{10}$)', fontsize=13)
ax.set_ylabel(f'K score', fontsize=13)
ax.set_title('NTK条件数与K score的互补性分析\n(DARTS搜索空间, CIFAR-10)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right', framealpha=0.9,
          edgecolor='gray', fancybox=True)
ax.grid(True, alpha=0.25, which='both', linestyle='--')

plt.tight_layout()
scatter_path = os.path.join(SAVE_DIR, 'ntk_vs_kscore_scatter.png')
plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  保存: {scatter_path}")

# ============================================================
# 图2: 箱线图对比 (NTK用对数y轴; K score用相对值)
# ============================================================
print("绘制箱线图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ntk_data = []
k_data = []
tick_labels_list = []
for cat_key in order:
    results = raw_data[cat_key]
    tick_labels_list.append(short_labels[cat_key])
    ntk_data.append([r['ntk_score'] for r in results])
    k_data.append([r['k_score'] - K_OFFSET for r in results])

# NTK箱线图 (对数y轴)
bp1 = axes[0].boxplot(ntk_data, tick_labels=tick_labels_list,
                      patch_artist=True, widths=0.5,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp1['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_yscale('log')
axes[0].set_ylabel(r'NTK条件数 ($\log_{10}$ 坐标)', fontsize=12)
axes[0].set_title('各类架构的NTK条件数分布\n(越低 → 可训练性越强)', fontsize=12)
axes[0].grid(True, alpha=0.25, axis='y', which='both', linestyle='--')
axes[0].tick_params(axis='x', labelsize=12)

# K score箱线图 (相对值)
bp2 = axes[1].boxplot(k_data, tick_labels=tick_labels_list,
                      patch_artist=True, widths=0.5,
                      showfliers=True,
                      flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp2['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_ylabel(f'K score − {K_OFFSET}', fontsize=12)
axes[1].set_title('各类架构的K score分布\n(越高 → 表达能力越强)', fontsize=12)
axes[1].grid(True, alpha=0.25, axis='y', linestyle='--')
axes[1].tick_params(axis='x', labelsize=12)

# K score y轴范围
all_k_rel = []
for results in raw_data.values():
    all_k_rel.extend([r['k_score'] - K_OFFSET for r in results])
k_min, k_max = min(all_k_rel), max(all_k_rel)
k_margin = (k_max - k_min) * 0.05
axes[1].set_ylim(k_min - k_margin, k_max + k_margin)

plt.suptitle('NTK条件数与K score在不同操作偏好架构上的互补特性',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
boxplot_path = os.path.join(SAVE_DIR, 'boxplot_comparison.png')
plt.savefig(boxplot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  保存: {boxplot_path}")

# ============================================================
# 图3: 柱状图 (NTK中位数+对数y轴; K score均值用相对值)
# ============================================================
print("绘制柱状图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_labels_bar = ['无参数操作', '均衡混合', '有参数操作']
x = np.arange(len(x_labels_bar))
width = 0.4

k_means_list = []
k_stds_list = []
ntk_medians = []
for cat_key in order:
    results = raw_data[cat_key]
    ntk_vals = [r['ntk_score'] for r in results]
    k_vals = [r['k_score'] - K_OFFSET for r in results]
    ntk_medians.append(np.median(ntk_vals))
    k_means_list.append(np.mean(k_vals))
    k_stds_list.append(np.std(k_vals))

# NTK柱状图 - 中位数 + 对数y轴
bars1 = axes[0].bar(x, ntk_medians, width,
                    color=box_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars1, ntk_medians):
    label_text = f'{val:.1f}' if val < 1e6 else f'{val:.2e}'
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels_bar, fontsize=11)
axes[0].set_ylabel(r'NTK条件数 (中位数, $\log_{10}$)', fontsize=11)
axes[0].set_title('NTK条件数对比 (越低越好)', fontsize=12, fontweight='bold')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.25, axis='y', which='both', linestyle='--')

# K score柱状图 (相对值)
bars2 = axes[1].bar(x, k_means_list, width, yerr=k_stds_list, capsize=5,
                    color=box_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars2, k_means_list):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels_bar, fontsize=11)
axes[1].set_ylabel(f'K score − {K_OFFSET} (均值±标准差)', fontsize=11)
axes[1].set_title('K score对比 (越高越好)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.25, axis='y', linestyle='--')

k_bar_min = min(k_means_list) - max(k_stds_list) - 5
axes[1].set_ylim(bottom=k_bar_min)

plt.tight_layout()
bar_path = os.path.join(SAVE_DIR, 'bar_comparison.png')
plt.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  保存: {bar_path}")

print(f"\n所有图表已保存到: {SAVE_DIR}")
