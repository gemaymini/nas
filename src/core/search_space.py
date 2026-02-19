import random 
from typing import List, Optional
from configuration.config import config
from core.encoding import CellEncoding, Individual, Edge

class SearchSpace:
    def __init__(self):
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        self.operations = config.OPERATIONS
        self.num_operations = len(self.operations)
        
    def get_valid_sources(self, node_idx: int) -> List[int]:
        return list(range(2 + node_idx))
    
    def sample_edge(self, node_idx: int) -> Edge:
        valid_sources = self.get_valid_sources(node_idx)
        source = random.choice(valid_sources)
        op_id = random.randint(0,self.num_operations - 1)
        return Edge(source= source, op_id= op_id)
    
    def sample_cell(self) -> CellEncoding:
        edges = []
        for node_idx in range(self.num_nodes):
            node_edges = [self.sample_edge(node_idx) for _ in range(self.edges_per_node)]
            edges.append(node_edges)
        return CellEncoding(edges= edges)
    
    def sample_individual(self) -> Individual:
        normal_cell = self.sample_cell()
        reduction_cell = self.sample_cell()
        return Individual(normal_cell= normal_cell, reduction_cell= reduction_cell)
        
search_space = SearchSpace()
