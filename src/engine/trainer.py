import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from typing import Tuple, List
from configuration.config import config
from utils.logger import logger

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """Computes the accuracy over the top-k predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

class NetworkTrainer:
    def __init__(self):
        self.device = 'cuda'
        
    def train_one_epoch(self, model: nn.Module, trainloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                        total_epochs: int, auxiliary_weight: float = 0.0) -> Tuple[float, float, float]:
        model.train()
        running_loss = 0.0
        top1_sum = 0.0
        top5_sum = 0.0
        total_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                logits, logits_aux = outputs
                loss = criterion(logits, targets)
                if logits_aux is not None:
                    loss_aux = criterion(logits_aux, targets)
                    loss += auxiliary_weight * loss_aux
            else:
                logits = outputs
                loss = criterion(logits, targets)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            
            running_loss += loss.item()
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            top1_sum += prec1
            top5_sum += prec5
            total_batches += 1
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(trainloader):
                progress = (batch_idx + 1) / len(trainloader) * 100
                avg_top1 = top1_sum / total_batches
                avg_top5 = top5_sum / total_batches
                print(f'\r [Epoch {epoch}/{total_epochs}] Batch: {batch_idx + 1}/{len(trainloader)}'
                      f'({progress:.1f}%) | Loss: {running_loss/(batch_idx + 1):.4f} | '
                      f'Top1: {avg_top1:.2f}% | Top5: {avg_top5:.2f}%',
                      end='', flush=True)
        print()
        avg_loss = running_loss / len(trainloader)
        avg_top1 = top1_sum / total_batches
        avg_top5 = top5_sum / total_batches
        return avg_loss, avg_top1, avg_top5
    
    def evaluate(self, model: nn.Module, testloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        model.eval()
        test_loss = 0.0
        top1_sum = 0.0
        top5_sum = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top1_sum += prec1
                top5_sum += prec5
                total_batches += 1
        
        avg_loss = test_loss / len(testloader)
        avg_top1 = top1_sum / total_batches
        avg_top5 = top5_sum / total_batches
        return avg_loss, avg_top1, avg_top5
    
    def train_network(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader, full_train: bool = True) -> Tuple[float, List[dict]]:
        if full_train:
            epochs = config.FULL_TRAIN_EPOCHS
        else:
            epochs = config.SHORT_TRAIN_EPOCHS
        lr = config.LEARNING_RATE
        momentum = config.MOMENTUM
        weight_decay = config.WEIGHT_DECAY
        patience = config.EARLY_STOP_PATIENCE
        min_delta = config.EARLY_STOP_MIN_DELTA
        torch.cuda.empty_cache()
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr= lr, momentum= momentum, weight_decay= weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= epochs)
        auxiliary_weight = config.AUXILIARY_WEIGHT if full_train else 0.0
        
        history = []
        best_top1 = 0.0
        best_top5 = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        
        try:
            for epoch in range(1, epochs + 1):
                # Learning rate warmup for ImageNet
                if epoch <= config.LR_WARMUP_EPOCHS:
                    warmup_lr = lr * epoch / config.LR_WARMUP_EPOCHS
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                
                if full_train:
                    drop_prob = config.DROP_PATH_MAX * epoch / epochs
                    model.update_drop_path_prob(drop_prob)
                train_loss, train_top1, train_top5 = self.train_one_epoch(model, trainloader, criterion, optimizer, epoch, epochs, auxiliary_weight)
                test_loss, test_top1, test_top5 = self.evaluate(model, testloader, criterion)
                
                # Only step scheduler after warmup completes
                if epoch > config.LR_WARMUP_EPOCHS:
                    scheduler.step()
                
                train_top1_err = 100.0 - train_top1
                train_top5_err = 100.0 - train_top5
                test_top1_err = 100.0 - test_top1
                test_top5_err = 100.0 - test_top5
                
                history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_top1': train_top1,
                    'train_top5': train_top5,
                    'train_top1_err': train_top1_err,
                    'train_top5_err': train_top5_err,
                    'test_loss': test_loss,
                    'test_top1': test_top1,
                    'test_top5': test_top5,
                    'test_top1_err': test_top1_err,
                    'test_top5_err': test_top5_err,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                if test_top1 > best_top1 + min_delta:
                    best_top1 = test_top1
                    best_top5 = test_top5
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else: 
                    epochs_without_improvement += 1
                print(f' [Epoch {epoch}/{epochs}] '
                      f'Train Top1 Err: {train_top1_err:.2f}% | '
                      f'Test Top1 Err: {test_top1_err:.2f}% | Test Top5 Err: {test_top5_err:.2f}% | '
                      f'Best Top1: {best_top1:.2f}% | '
                      f'No Improve: {epochs_without_improvement}/{patience}')
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}."
                                f"No improvement > {min_delta}% for {patience} epochs.")
                    print(f" *** Early stopping at epoch {epoch} ***")
                    break
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("OOM during training. Clearing cache.")
            raise e
        finally:
            torch.cuda.empty_cache()
            
        model.load_state_dict(best_model_wts)
        return best_top1, best_top5, history