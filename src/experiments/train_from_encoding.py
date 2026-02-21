"""
从 encoding.json 加载架构编码，执行完整训练。

用法:
    uv run python -m experiments.train_from_encoding --json <path_to_encoding.json>

encoding.json 格式示例:
{
    "normal_cell": [0, 2, 1, 3, 0, 4, 1, 5, ...],
    "reduction_cell": [0, 2, 1, 3, 0, 4, 1, 5, ...]
}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import torch
import numpy as np
import random

from configuration.config import config
from core.encoding import CellEncoding, Individual
from engine.evaluator import FinalEvaluator
from utils.logger import logger


def load_encoding(json_path: str) -> Individual:
    """从 JSON 文件加载架构编码并构建 Individual"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'normal_cell' not in data or 'reduction_cell' not in data:
        raise ValueError("JSON 文件必须包含 'normal_cell' 和 'reduction_cell' 字段")
    
    normal_cell = CellEncoding.from_list(data['normal_cell'])
    reduction_cell = CellEncoding.from_list(data['reduction_cell'])
    individual = Individual(normal_cell=normal_cell, reduction_cell=reduction_cell)
    
    return individual


def main():
    parser = argparse.ArgumentParser(description='从 encoding.json 加载架构并执行完整训练')
    parser.add_argument('--json', type=str, default="src/experiments/encoding.json",
                        help='encoding.json 文件路径')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (默认使用 config 中的种子)')
    args = parser.parse_args()
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed: {seed}")
    
    # 加载架构
    logger.info(f"Loading encoding from: {args.json}")
    individual = load_encoding(args.json)
    individual.print_architecture()
    
    # 完整训练
    logger.info(f"Starting full training for {config.FULL_TRAIN_EPOCHS} epochs...")
    logger.info(f"Dataset: {config.FINAL_DATASET} | Init Channels: {config.INIT_CHANNELS} | "
                f"Cells per Stage: {config.CELLS_PER_STAGE}")
    
    evaluator = FinalEvaluator()
    best_acc, result = evaluator.evaluate_individual(individual, full_train=True)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  Best Accuracy:  {best_acc:.2f}%")
    print(f"  Param Count:    {result['param_count']:,}")
    print(f"  Train Time:     {result['train_time']:.1f}s")
    print(f"  Genotype:       {result['genotype']}")
    print(f"  Model saved to: {result['model_path']}")
    if result.get('plot_path'):
        print(f"  Plot saved to:  {result['plot_path']}")
    print("=" * 60)
    
    logger.info(f"Full training complete. Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
