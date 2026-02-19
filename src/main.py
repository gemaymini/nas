import argparse
import torch
import sys
import os
import random
import numpy as np
from configuration.config import config
from search.evolution import NSGA2NAS
from utils.logger import logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def parse_args():
    parser = argparse.ArgumentParser(description='NSGA-II Multi-Objective NAS')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    parser.add_argument('--no_final_eval', action='store_true')
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()

def main():
    logger.setup_file_logging()
    
    args = parse_args()
    set_seed(args.seed)
    
    logger.info(f'Dataset: {config.FINAL_DATASET}, Num Classes: {config.NUM_CLASSES}')
    logger.info(f'cuda_is_available: {torch.cuda.is_available()}')
    logger.info(f'Algorithm: NSGA-II Multi-Objective (NTK + K)')
    
    nas = NSGA2NAS()
    
    if args.resume:
        try: 
            nas.load_checkpoint(args.resume)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
            
    try:
        nas.run_search()
        if not args.no_final_eval:
            nas.run_screening_and_training()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        nas._save_checkpoint()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        nas._save_checkpoint()
        raise
    
if __name__ == '__main__':
    main()