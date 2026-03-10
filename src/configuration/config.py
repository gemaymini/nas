import os
import random

class Config:
    def __init__(self):
        self.MODE = 'PRO'
        
        if self.MODE == 'DEV':
            self.POPULATION_SIZE = 100
            self.MAX_GEN = 20
            
            self.HISTORY_TOP_N1 = 5
            self.SHORT_TRAIN_EPOCHS = 1
            self.HISTORY_TOP_N2 = 1
            self.FULL_TRAIN_EPOCHS = 1
            
            self.TRAIN_BATCH_SIZE = 64
            self.EVAL_BATCH_SIZE = 64
            
        elif self.MODE == 'PRO':
            self.POPULATION_SIZE = 100
            self.MAX_GEN = 30
            
            self.HISTORY_TOP_N1 = 20
            self.SHORT_TRAIN_EPOCHS = 12
            self.HISTORY_TOP_N2 = 3
            self.FULL_TRAIN_EPOCHS = 250
            
            self.TRAIN_BATCH_SIZE = 96
            self.EVAL_BATCH_SIZE = 256
        
        self.TOURNAMENT_SIZE = 5
        self.MUTATION_PROB = 0.8
        self.MUTATION_TIME_PROB = [1, 1, 1, 2, 2, 3]
        self.MUTATION_CELL_PROB = 0.5
        self.NUM_WORKERS = 8
        self.MOMENTUM = 0.9 
        self.WEIGHT_DECAY = 3e-4
        self.AUXILIARY_WEIGHT = 0.4
        self.DROP_PATH_MAX = 0.3
        self.GRAD_CLIP = 5.0
        self.FINAL_DATASET = "imagenet"
        if self.FINAL_DATASET == 'cifar10':
            self.NUM_CLASSES = 10
        elif self.FINAL_DATASET == 'cifar100':
            self.NUM_CLASSES = 100
        elif self.FINAL_DATASET == 'imagenet':
            self.NUM_CLASSES = 1000
            
            
        self.NUM_NODES = 4
        self.EDGES_PER_NODE = 2
        
        # DARTS paper: CIFAR 20 cells / 36 channels, ImageNet 14 cells / 48 channels
        if self.FINAL_DATASET == 'imagenet':
            self.INIT_CHANNELS = 48
            self.CELLS_PER_STAGE = 4        # total = 3*(4+1)-1 = 14 cells
            self.LEARNING_RATE = 0.1        # DARTS ImageNet: 0.1
            self.EARLY_STOP_PATIENCE = 150
            self.EARLY_STOP_MIN_DELTA = 0.01
            self.DROP_PATH_MAX = 0.0        # DARTS ImageNet: no drop path
            self.LR_WARMUP_EPOCHS = 5       # linear warmup for ImageNet
        else:
            self.INIT_CHANNELS = 36
            self.CELLS_PER_STAGE = 6        # total = 3*(6+1)-1 = 20 cells
            self.LEARNING_RATE = 0.025      # DARTS CIFAR: 0.025
            self.EARLY_STOP_PATIENCE = 75
            self.EARLY_STOP_MIN_DELTA = 0.01
            self.LR_WARMUP_EPOCHS = 0
            
        self.NUM_STAGES = 3
        
        self.OPERATIONS = [
            "zero",
            "skip_connect",
            "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "dil_conv_5x5",
            "max_pool_3x3",
            "avg_pool_3x3"
        ]
        
        self.NTK_NUM_BATCH = 1
        self.NTK_RECAL_NUM_BATCH = 1
        self.K_NUM_BATCH = 1
        self.K_EPS = 1e-6
        
        self.NTK_VALID_THRESHOLD = 5000
        self.NTK_FAIL_SCORE = 5000
        self.K_FAIL_SCORE = -5000
        
        self.IMAGENET_DATA_ROOT = '/root/autodl-tmp/imagenet'
        self.DATA_ROOT = './data' 
        self.LOG_DIR = './logs'
        self.LOG_LEVEL = 'INFO'
        self.CHECKPOINT_DIR = './checkpoints'
        self.RANDOM_SEED = random.randint(0, 2**32 - 1)
        
config = Config()