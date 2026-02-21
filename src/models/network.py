import torch
import torch.nn as nn
from configuration.config import config
from core.encoding import CellEncoding
from models.operator import get_op, FactorizedReduce, ReLUConvBN, DropPath

class Cell(nn.Module):
    def __init__(self, cell_encoding: CellEncoding, C_prev_prev: int, C_prev: int, C: int, reduction: bool, reduction_prev: bool):
        super().__init__()
        self.num_nodes = config.NUM_NODES
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        self._ops = nn.ModuleList()
        self._drop_paths = nn.ModuleList()
        self._edge_indices = []
        stride = 2 if reduction else 1
        for node_idx, node_edges in enumerate(cell_encoding.edges):
            for edge_idx, edge in enumerate(node_edges):
                op_name = config.OPERATIONS[edge.op_id]
                if edge.source < 2 and reduction:
                    op_stride = stride
                else: 
                    op_stride = 1
                op = get_op(op_name, C, op_stride)
                self._ops.append(op)
                self._drop_paths.append(DropPath())
                self._edge_indices.append((node_idx, edge_idx, edge.source))
        self.out_channels = self.num_nodes * C
        self.cell_encoding = cell_encoding
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        op_idx = 0
        for _ in range(self.num_nodes):
            node_inputs = []
            edges_per_node = config.EDGES_PER_NODE
            
            for _ in range(edges_per_node):
                _, _, source = self._edge_indices[op_idx] 
                h = states[source]
                h = self._ops[op_idx](h)
                h = self._drop_paths[op_idx](h)
                node_inputs.append(h)
                op_idx += 1
            
            node_output = sum(node_inputs)
            states.append(node_output)
    
        return torch.cat(states[2:],dim= 1)
    
class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, config.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Network(nn.Module):
    def __init__(self, normal_cell: CellEncoding, reduction_cell: CellEncoding, auxiliary: bool = False):
        super().__init__()
        self.num_classes = config.NUM_CLASSES
        self.init_channels = config.INIT_CHANNELS
        self.auxiliary = auxiliary
        self.cells_per_stage = config.CELLS_PER_STAGE
        self.num_stages = config.NUM_STAGES
        
        C = self.init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * 3, 3, padding= 1, bias= False),
            nn.BatchNorm2d(C * 3)
        )
        
        self.cells = nn.ModuleList()
        C_prev_prev = C * 3
        C_prev = C * 3
        C_curr = C
        reduction_prev = False
        
        cell_idx = 0
        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                reduction = True
                cell = Cell(reduction_cell, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                self.cells.append(cell)
                
                C_prev_prev = C_prev
                C_prev = cell.out_channels
                C_curr *= 2
                reduction_prev = True
                cell_idx += 1
                
                if self.auxiliary and stage_idx == self.num_stages - 1:
                    self.aux_cell_idx = len(self.cells) - 1
                    self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev)
            
            for _ in range(self.cells_per_stage):
                reduction = False
                cell = Cell(normal_cell, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
                self.cells.append(cell)
                
                C_prev_prev = C_prev
                C_prev = cell.out_channels
                reduction_prev = False
                cell_idx += 1
                
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        
        self.normal_cell_encoding = normal_cell
        self.reduction_cell_encoding = reduction_cell
        
    def forward(self, x: torch.Tensor):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self.auxiliary and self.training:
                if i == self.aux_cell_idx:
                    logits_aux = self.auxiliary_head(s1)
            
        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        if self.auxiliary and self.training:
            return out, logits_aux
        else: 
            return out
    
    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def update_drop_path_prob(self, drop_prob: float):
        """Update drop path probability for all DropPath layers in the network."""
        for cell in self.cells:
            for dp in cell._drop_paths:
                dp.drop_prob = drop_prob