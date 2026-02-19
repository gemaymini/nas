import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from typing import Tuple, List
from configuration.config import config
from utils.logger import logger

class NetworkTrainer:
    def __init__(self):
        self.device = 'cuda'
        
    def train_one_epoch(self, model: nn.Module, trainloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                        total_epochs: int, auxiliary_weight: float = 0.0) -> Tuple[float, float]:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(trainloader):
                progress = (batch_idx + 1) / len(trainloader) * 100
                acc = 100. * correct / total
                print(f'\r [Epoch {epoch}/{total_epochs}] Batch: {batch_idx + 1}/{len(trainloader)}'
                      f'({progress:.1f}%) | Loss: {running_loss/(batch_idx + 1):.4f} | Acc: {acc:.2f}%',
                      end='', flush=True)
        print()
        avg_loss = running_loss / len(trainloader)
        accurary = 100. * correct / total
        return avg_loss, accurary
    
    def evaluate(self, model: nn.Module, testloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = test_loss / len(testloader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
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
        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        
        try:
            for epoch in range(1, epochs + 1):
                train_loss, train_acc = self.train_one_epoch(model, trainloader, criterion, optimizer, epoch, epochs, auxiliary_weight)
                test_loss, test_acc = self.evaluate(model, testloader, criterion)
                scheduler.step()
                history.append({
                    'epoch':epoch,
                    'train_loss':train_loss,
                    'train_acc':train_acc,
                    'test_loss':test_loss,
                    'test_acc':test_acc,
                    'lr':optimizer.param_groups[0]['lr']
                })
                
                if test_acc > best_acc +min_delta:
                    best_acc = test_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else: 
                    epochs_without_improvement += 1
                print(f' [Epoch {epoch}/{epochs}] Train Acc: {train_acc:.2f}% |'
                      f'Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% |'
                      f'No Improve: {epochs_without_improvement}/{patience}')
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered ad epoch {epoch}."
                                f"No improvement > {min_delta}% for {patience} epochs.")
                    print(f"' *** Early stopping at epoch {epoch} ***")
                    break
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("OOM during training. Clearing cache.")
            raise e
        finally:
            torch.cuda.empty_cache()
            
        model.load_state_dict(best_model_wts)
        return best_acc, history