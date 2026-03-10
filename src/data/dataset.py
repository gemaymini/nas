import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from configuration.config import config

class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
        
        
    def __call__(self,img: torch.Tensor) -> torch.Tensor:
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h,w),np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img
    
class DatasetLoader:
    def __init__(self):
        self.root = config.DATA_ROOT
        if os.path.exists(os.path.join(self.root, 'cifar-10-batches-py')):
            self.cifar10_download = False
        else: 
            self.cifar10_download = True
        if os.path.exists(os.path.join(self.root, 'cifar-100-python')):
            self.cifar100_download = False
        else: 
            self.cifar100_download = True

        self.imagenet_root = config.IMAGENET_DATA_ROOT
        self.train_batch_size = config.TRAIN_BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        self.dataset_name = config.FINAL_DATASET
        self.eval_batch_size = config.EVAL_BATCH_SIZE
        
            
    def get_cifar10(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                Cutout(1, 16)
            ]
        )
        trainsform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ]
        )
        
        trainset = torchvision.datasets.CIFAR10(
            root= self.root, train= True, download= self.cifar10_download, transform= transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size= self.train_batch_size, shuffle= True, num_workers= self.num_workers
        )
        
        testset = torchvision.datasets.CIFAR10(
            root= self.root, train= False, download= self.cifar10_download, transform= trainsform_test
        )
        testloader = DataLoader(
            testset, batch_size= self.train_batch_size, shuffle= False, num_workers= self.num_workers
        )
        return trainloader, testloader
    
    def get_cifar100(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
                Cutout(1, 16)
            ]
        )
        trainsform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root= self.root, train= True, download= self.cifar100_download, transform= transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size= self.train_batch_size, shuffle= True, num_workers= self.num_workers
        )
        
        testset = torchvision.datasets.CIFAR100(
            root= self.root, train= False, download= self.cifar100_download, transform= trainsform_test
        )
        testloader = DataLoader(
            testset, batch_size= self.train_batch_size, shuffle= False, num_workers= self.num_workers
        )
        
        return trainloader, testloader
    
    def get_imagenet(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )
        
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.imagenet_root, 'train'),
            transform=transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=self.train_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.imagenet_root, 'val'),
            transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=self.eval_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return trainloader, testloader
    
    def get_ntk_trainloader(self):
        if self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
            ])
            dataset = torchvision.datasets.CIFAR10(
                root= self.root, train= True, download= self.cifar10_download, transform= transform
            )
        elif self.dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761))
            ])
            dataset = torchvision.datasets.CIFAR100(
                root= self.root, train= True, download= self.cifar100_download, transform= transform
            )
        elif self.dataset_name == 'imagenet':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])
            dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.imagenet_root, 'train'),
                transform=transform
            )
            
        loader = DataLoader(
            dataset, batch_size= self.eval_batch_size, shuffle= False, pin_memory= True
        )
        
        return loader
    
    def get_dataset(self):
        if self.dataset_name == 'cifar10':
            return self.get_cifar10()
        elif self.dataset_name == 'cifar100':
            return self.get_cifar100()
        elif self.dataset_name == 'imagenet':
            return self.get_imagenet()
        
datasetloader = DatasetLoader()