from typing import List, Tuple
from dataclasses import dataclass
from configuration.config import config

@dataclass
class Edge:
    source: int
    op_id: int
    
    def to_list(self) -> List[int]:
        return [self.source, self.op_id]
    
    @classmethod
    def from_list(cls, data: List[int]) -> "Edge":
        return cls(source= data[0], op_id= data[1])
    
    def copy(self) -> "Edge":
        return Edge(source= self.source, op_id= self.op_id)
    
class CellEncoding:
    def __init__(self, edges: List[List[Edge]]):
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        self.edges = edges
        
    def to_list(self) -> List[int]:
        result = []
        for node_edges in self.edges:
            for edge in node_edges:
                result.extend(edge.to_list())
        return result
    
    @classmethod
    def from_list(cls, encoding: List[int]) -> "CellEncoding":
        num_nodes = config.NUM_NODES
        edges_per_node = config.EDGES_PER_NODE
        expected_length = num_nodes * edges_per_node * 2
        
        if len(encoding) != expected_length:
            raise ValueError(f"Expected encoding length {expected_length}, get {len(encoding)}")
        
        edges = []
        idx = 0
        
        for _ in range(num_nodes):
            node_edges = []
            for _ in range(edges_per_node):
                edge_data = encoding[idx:idx+2]
                node_edges.append(Edge.from_list(edge_data))
                idx += 2
            edges.append(node_edges)
        
        cell = cls.__new__(cls)
        cell.num_nodes = num_nodes
        cell.edges_per_node = edges_per_node
        cell.edges = edges
        return cell
    
    def copy(self) -> 'CellEncoding':
        new_edges = []
        for node_edges in self.edges:
            new_edges.append([edge.copy() for edge in node_edges])
            
        cell = CellEncoding.__new__(CellEncoding)
        cell.num_nodes = self.num_nodes
        cell.edges_per_node=self.edges_per_node
        cell.edges = new_edges
        return cell
    
    def get_edge(self, node_idx: int, edge_idx: int) -> Edge:
        return self.edges[node_idx][edge_idx]
    
    def set_edge(self, node_idx: int, edge_idx: int, edge: Edge):
        self.edges[node_idx][edge_idx] = edge
    
    def __repr__(self):
        lines = [f"CellEncoding(num_nodes={self.num_nodes}, edges_per_node={self.edges_per_node}):"]
        for node_idx, node_edges in enumerate(self.edges):
            edge_strs = []
            for edge in node_edges:
                op_name = config.OPERATIONS[edge.op_id] 
                source_name = f"Input_{edge.source}" if edge.source < 2 else f"Node_{edge.source - 2}"
                edge_strs.append(f"{source_name}->{op_name}")
            lines.append(f" Node_{node_idx}:{', '.join(edge_strs)}")
        return "\n".join(lines)
    
class Individual:
    _id_counter = 0
    def __init__(self, normal_cell: CellEncoding, reduction_cell: CellEncoding):
        Individual._id_counter += 1
        self.id = Individual._id_counter
        self.normal_cell = normal_cell
        self.reduction_cell = reduction_cell
        self.param_count = None
        self.accuracy = None
        self.ntk_score = None
        self.k_score = None
        self.rank = None
        self.combined_rank_score = None
        self.objectives = None
        self.domination_count = 0
        self.dominated_set = []
        
    def copy(self) -> "Individual":
        new_ind = Individual(normal_cell= self.normal_cell.copy(), reduction_cell= self.reduction_cell.copy())
        Individual._id_counter -= 1
        new_ind.id = self.id
        new_ind.param_count = self.param_count
        new_ind.accuracy = self.accuracy
        new_ind.ntk_score = self.ntk_score
        new_ind.k_score = self.k_score
        new_ind.rank = self.rank
        new_ind.combined_rank_score = self.combined_rank_score
        if self.objectives is not None:
            new_ind.objectives = list(self.objectives)
        return new_ind
    
    def reset_evaluation(self):
        self.param_count = None
        self.accuracy = None
        self.ntk_score = None
        self.k_score = None
        self.rank = None
        self.combined_rank_score = None
        self.objectives = None
        self.domination_count = 0
        self.dominated_set = []
        
    def dominates(self, other: "Individual") -> bool:
        if self.objectives is None or other.objectives is None:
            return False
        at_least_one_better = False
        for s, o in zip(self.objectives, other.objectives):
            if s > o:
                return False
            elif s < o:
                at_least_one_better = True
        return at_least_one_better
    
    @classmethod
    def update_id_counter(cls, max_id: int):
        if max_id >= cls._id_counter:
            cls._id_counter = max_id + 1
            
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'normal_cell': self.normal_cell.to_list(),
            'reduction_cell': self.reduction_cell.to_list(),
            'param_count': self.param_count,
            'accuracy': self.accuracy,
            'ntk_score': self.ntk_score,
            'k_score': self.k_score,
            'rank': self.rank,
            'combined_rank_score': self.combined_rank_score,
            'objectives': self.objectives,
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Individual':
        ind = cls(
            normal_cell= CellEncoding.from_list(data['normal_cell']),
            reduction_cell= CellEncoding.from_list(data['reduction_cell'])
        )
        ind.id = data.get('id', ind.id)
        ind.param_count = data.get('param_count')
        ind.accuracy = data.get('accuracy')
        ind.ntk_score = data.get('ntk_score')
        ind.k_score = data.get('k_score')
        ind.rank = data.get('rank')
        ind.combined_rank_score = data.get('combined_rank_score')
        ind.objectives = data.get('objectives')
        return ind

    def __repr__(self):
        return (f'Individual(id={self.id}, rank={self.rank}, '
                f'ntk={self.ntk_score}, k={self.k_score}, '
                f'params={self.param_count}, acc={self.accuracy})\n'
                f'normal-cell={self.normal_cell}\n'
                f'reduction-cell={self.reduction_cell}\n')
    
    def print_architecture(self):
        print(f"\n{'='*60}")
        print(f"Individual {self.id}")
        print(f"{'='*60}")
        print(f"Rank: {self.rank}")
        print(f"NTK Score: {self.ntk_score}")
        print(f"K Score: {self.k_score}")
        print(f"Objectives: {self.objectives}")
        print(f"Combined Rank Score: {self.combined_rank_score}")
        print(f"Param Count: {self.param_count}")
        print(f"Accuracy: {self.accuracy}")
        print(f"\n--- Normal Cell ---")
        print(self.normal_cell)
        print(f"\n--- Reduction Cell ---")
        print(self.reduction_cell)
        print(f"\n{'='*60}\n")
        
    def get_genotype(self) -> dict:
        def cell_to_genotype(cell: CellEncoding) -> List[Tuple[str, int]]:
            genotype = []
            for _, node_edges in enumerate(cell.edges):
                for edge in node_edges:
                    op_name = config.OPERATIONS[edge.op_id]
                    genotype.append((op_name, edge.source))
            return genotype
        
        return {
            'normal': cell_to_genotype(self.normal_cell),
            'normal_concat': list(range(2, 2 + config.NUM_NODES)),
            'reduce': cell_to_genotype(self.reduction_cell),
            'reduce_concat': list(range(2, 2 + config.NUM_NODES))
        }
    
