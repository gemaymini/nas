import torch
import torch.nn as nn
import numpy as np
import gc
import time
import os
import json
import matplotlib.pyplot as plt
from typing import Tuple, List
from configuration.config import config
from core.encoding import Individual
from models.network import Network
from utils.logger import logger
from data.dataset import datasetloader
from engine.trainer import NetworkTrainer

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
class NTKEvaluator:
    def __init__(self):
        self.num_classes = config.NUM_CLASSES
        self.recal_num_batch = config.NTK_RECAL_NUM_BATCH
        self.ntk_num_batch = config.NTK_NUM_BATCH
        self.k_num_batch = config.K_NUM_BATCH
        self.k_eps = config.K_EPS
        self.device = 'cuda'
        self.trainloader = datasetloader.get_ntk_trainloader()
    
    def recal_bn(self, network: nn.Module):
        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.data.fill_(0)
                m.running_var.data.fill_(0)
                m.num_batches_tracked.data.zero_()
                m.momentum = None
                
        network.train()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.trainloader):
                if i >= self.recal_num_batch:
                    break
                inputs = inputs.to(device=self.device, non_blocking=True)
                _ = network(inputs)
        return network

    def _compute_ntk_eigenvalues(self, network: nn.Module):
        network.train()
                
        grads = []
        for i, (inputs, targets) in enumerate(self.trainloader):
            if self.ntk_num_batch <= i:
                break
            inputs = inputs.to(device=self.device, non_blocking=True)
            network.zero_grad()
            inputs_ = inputs.clone().to(device=self.device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]
            
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
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
                logger.warning("NaN detected in eigenvalues")
                return None
                
            if eigenvalues[-1].item() < 1e-8:
                logger.warning(f"Dead network detected (Max Eigenvalue < 1e-8: {eigenvalues[-1].item():.6e}).")
                return None
            
            return eigenvalues
        except Exception as e:
            logger.warning(f'NTK eigenvalue computation failed: {e}')
            return None

    def _compute_k_logdet(self, network: nn.Module) -> float:
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
                for i, (inputs, _) in enumerate(self.trainloader):
                    if i >= self.k_num_batch:
                        break
                    inputs = inputs.to(device=self.device, non_blocking=True)
                    _ = network(inputs)

            if k_holder['K'] is None:
                return None

            k = k_holder['K']
            k = k + self.k_eps * torch.eye(k.size(0), device=k.device, dtype=k.dtype)
            sign, logdet = torch.linalg.slogdet(k.double())
            if sign <= 0:
                logger.warning('K logdet sign <= 0')
                return None
            return logdet.item()
        except Exception as e:
            logger.warning(f'K computation failed: {e}')
            return None
        finally:
            for h in handles:
                h.remove()
    
    def compute_ntk_score(self, network: nn.Module) -> float:
        try:
            network = network.to(self.device)    
            if self.recal_num_batch > 0:
                network = self.recal_bn(network)
            
            eigenvalues = self._compute_ntk_eigenvalues(network)
            if eigenvalues is None:
                return config.NTK_FAIL_SCORE
        
            condition_number = (eigenvalues[-1] / eigenvalues[0]).item()
            
            if np.isnan(condition_number) or np.isinf(condition_number) or condition_number > config.NTK_FAIL_SCORE:
                return config.NTK_FAIL_SCORE
            
            return condition_number
        except Exception as e:
            logger.error(f"NTK computation failed: {e}")
            clear_gpu_memory()
            return config.NTK_FAIL_SCORE

    def compute_k_score(self, network: nn.Module) -> float:
        try:
            logdet = self._compute_k_logdet(network)
            if logdet is None:
                return config.K_FAIL_SCORE
            return logdet
        except Exception as e:
            logger.error(f"K computation failed: {e}")
            clear_gpu_memory()
            return config.K_FAIL_SCORE

    def evaluate_individual(self, individual: Individual) -> Individual:
        try:
            network = Network(individual.normal_cell, individual.reduction_cell)
            individual.param_count = network.get_param_count()
            
            ntk_score = self.compute_ntk_score(network)
            individual.ntk_score = ntk_score
            
            k_score = self.compute_k_score(network)
            individual.k_score = k_score
            
            if ntk_score >= config.NTK_VALID_THRESHOLD:
                logger.warning(
                    f"Individual {individual.id} marked invalid: "
                    f"NTK={ntk_score:.4e} >= threshold={config.NTK_VALID_THRESHOLD}"
                )
                individual.objectives = [config.NTK_FAIL_SCORE, -config.K_FAIL_SCORE]
            else:
                individual.objectives = [ntk_score, -k_score]
            
            del network
            clear_gpu_memory()
            
            logger.info(
                f"Eval {individual.id} | NTK={ntk_score:.4f} | "
                f"K={k_score:.4f} | Params={individual.param_count}"
            )
            return individual
        except Exception as e:
            logger.error(f"Failed to evaluate individual {individual.id}: {e}")
            individual.ntk_score = config.NTK_FAIL_SCORE
            individual.k_score = config.K_FAIL_SCORE
            individual.objectives = [config.NTK_FAIL_SCORE, -config.K_FAIL_SCORE]
            clear_gpu_memory()
            return individual
    
    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """评估整个种群"""
        logger.info(f"Evaluating population of {len(population)} individuals...")
        for i, ind in enumerate(population):
            self.evaluate_individual(ind)
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{len(population)}")
        logger.info(f"Population evaluation complete.")
        return population
        
class FinalEvaluator:
    def __init__(self):
        self.trainer = NetworkTrainer()
        self.trainloader, self.testloader = datasetloader.get_dataset()    
    
    def plot_training_history(self, history: list, individual_id: int, epochs: int, best_acc: float, param_count: int, save_dir: str):
        if not history:
            return
        
        epoch_list = [h['epoch'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        test_loss = [h['test_loss'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        test_acc = [h['test_acc'] for h in history]
        lr_list = [h['lr'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training History - Model {individual_id}\n'
                     f'Params: {param_count:,} | Best Test Acc: {best_acc:.2f}% | Epochs: {epochs}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Loss 曲线
        ax1 = axes[0, 0]
        ax1.plot(epoch_list, train_loss, 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epoch_list, test_loss, 'r-', label='Test Loss', linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Accuracy 曲线
        ax2 = axes[0, 1]
        ax2.plot(epoch_list, train_acc, 'b-', label='Train Acc', linewidth=1.5)
        ax2.plot(epoch_list, test_acc, 'r-', label='Test Acc', linewidth=1.5)
        ax2.axhline(y=best_acc, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Train/Test Acc 差距（过拟合指标）
        ax3 = axes[1, 0]
        gap = [train_acc[i] - test_acc[i] for i in range(len(train_acc))]
        ax3.plot(epoch_list, gap, 'purple', linewidth=1.5)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.fill_between(epoch_list, 0, gap, alpha=0.3, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train Acc - Test Acc (%)')
        ax3.set_title('Generalization Gap (Overfitting Indicator)')
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        ax4.plot(epoch_list, lr_list, 'green', linewidth=1.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f'training_curve_model_{individual_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Training curve saved to {plot_path}")
        return plot_path

    def evaluate_individual(self, individual: Individual, full_train: bool = True) -> Tuple[float, dict]:
        if full_train:
            epochs = config.FULL_TRAIN_EPOCHS
        else:
            epochs = config.SHORT_TRAIN_EPOCHS
        logger.info(f"Training individual {individual.id} for {epochs} epochs...")
        network = Network(individual.normal_cell, individual.reduction_cell, auxiliary=full_train)
        param_count = network.get_param_count()
        individual.param_count = param_count
        print(individual)

        start_time = time.time()
        best_acc, history = self.trainer.train_network(
            network, self.trainloader, self.testloader, full_train
        )
        train_time = time.time() - start_time

        if full_train:
            save_dir = os.path.join(config.CHECKPOINT_DIR, 'full_train_models')
        else:
            save_dir = os.path.join(config.CHECKPOINT_DIR, 'short_train_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{individual.id}_acc{best_acc:.2f}.pth')

        genotype = individual.get_genotype()
        save_dict = {
            'state_dict': network.state_dict(),
            'genotype': genotype,
            'normal_cell': individual.normal_cell.to_list(),
            'reduction_cell': individual.reduction_cell.to_list(),
            'accuracy': best_acc,
            'param_count': param_count,
            'history': history
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved model to {save_path}")
        logger.info(f"Model {individual.id} Genotype: {genotype}")

        # 短训时保存两种格式的编码到 JSON 文件
        if not full_train:
            encoding_save_path = save_path.replace('.pth', '_encoding.json')
            encoding_data = {
                'individual_id': individual.id,
                'accuracy': best_acc,
                'param_count': param_count,
                'genotype': genotype,
                'normal_cell_readable': individual.normal_cell.__repr__(),
                'reduction_cell_readable': individual.reduction_cell.__repr__(),
                'normal_cell': individual.normal_cell.to_list(),
                'reduction_cell': individual.reduction_cell.to_list(),
            }
            with open(encoding_save_path, 'w', encoding='utf-8') as f:
                json.dump(encoding_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved encoding to {encoding_save_path}")

        if full_train:
            plot_dir = os.path.join(config.LOG_DIR, 'training_curves', 'full_train')
        else:
            plot_dir = os.path.join(config.LOG_DIR, 'training_curves', 'short_train')
        try:
            plot_path = self.plot_training_history(
                history=history,
                individual_id=individual.id,
                epochs=epochs,
                best_acc=best_acc,
                param_count=param_count,
                save_dir=plot_dir
            )
        except Exception as e:
            logger.warning(f"Failed to generate training curve plot: {e}")
            plot_path = None

        result = {
            'individual_id': individual.id,
            'param_count': param_count,
            'best_accuracy': best_acc,
            'train_time': train_time,
            'history': history,
            'genotype': genotype,
            'model_path': save_path,
            'plot_path': plot_path
        }
        return best_acc, result

        
ntk_evaluator = NTKEvaluator()
final_evaluator = FinalEvaluator()
