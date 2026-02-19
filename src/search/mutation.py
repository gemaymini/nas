import random
import copy
from typing import List, Tuple
from configuration.config import config
from core.encoding import CellEncoding, Individual, Edge
from core.search_space import search_space

class MutationOperator:
    def __init__(self):
        self.num_nodes = config.NUM_NODES
        self.edges_per_node = config.EDGES_PER_NODE
        self.num_operations = len(config.OPERATIONS)
        
    def mutation_edge_operation(self, cell: CellEncoding) -> CellEncoding:
        node_idx = random.randint(0, self.num_nodes - 1)
        edge_idx = random.randint(0, self.edges_per_node - 1)
        edge = cell.get_edge(node_idx, edge_idx)
        
        valid_op_id = [i for i in range(self.num_operations) if i != edge.op_id]
        new_op_id = random.choice(valid_op_id)
        new_edge = Edge(source=edge.source, op_id=new_op_id)
        
        cell.set_edge(node_idx, edge_idx, new_edge)
        return cell
    
    def mutate_edge_source(self, cell: CellEncoding) -> CellEncoding:
        node_idx = random.randint(0, self.num_nodes - 1)
        edge_idx = random.randint(0, self.edges_per_node - 1)
        edge = cell.get_edge(node_idx, edge_idx)
        
        valid_sources = [i for i in range(node_idx + 2) if i != edge.source]
        new_source = random.choice(valid_sources)
        new_edge = Edge(source=new_source, op_id=edge.op_id)
        
        cell.set_edge(node_idx, edge_idx, new_edge)
        return cell
    
    def mutate_cell(self, cell: CellEncoding) -> CellEncoding:
        mutation_op_times = random.choices(config.MUTATION_TIME_PROB, k=1)[0]
        mutation_source_times = random.choices(config.MUTATION_TIME_PROB, k=1)[0]
        for _ in range(mutation_op_times):
            cell = self.mutation_edge_operation(cell)
        for _ in range(mutation_source_times):
            cell = self.mutate_edge_source(cell)
        return cell
    
    def mutate(self, individual: Individual) -> Individual:
        if random.random() < config.MUTATION_CELL_PROB:
            individual.normal_cell = self.mutate_cell(individual.normal_cell)
        if random.random() < config.MUTATION_CELL_PROB:
            individual.reduction_cell = self.mutate_cell(individual.reduction_cell)
        individual.reset_evaluation()
        return individual
    
class CrossoverOperator:
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        if random.random() < 0.5:
            return Individual(
                normal_cell=parent1.normal_cell.copy(),
                reduction_cell= parent2.reduction_cell.copy()
            )
        else:
            return Individual(
                normal_cell=parent2.normal_cell.copy(),
                reduction_cell= parent1.reduction_cell.copy()
            )

class NSGA2Selector:
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        for ind in population:
            ind.domination_count = 0
            ind.dominated_set = []
        
        fronts = [[]]
        
        for p in population:
            for q in population:
                if p is q:
                    continue
                if p.dominates(q):
                    p.dominated_set.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        # 移除最后一个空的前沿
        if not fronts[-1]:
            fronts.pop()
        
        return fronts

    def crowding_distance_assignment(self, front: List[Individual]):
        n = len(front)
        if n == 0:
            return
        
        for ind in front:
            ind.crowding_distance = 0.0
        
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        num_objectives = len(front[0].objectives)
        
        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.objectives[m])
            
            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            obj_range = obj_max - obj_min
            
            # 边界个体赋予无穷大距离
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                front[i].crowding_distance += (
                    (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                )
    
    def crowded_comparison(self, ind1: Individual, ind2: Individual) -> Individual:
        if ind1.rank < ind2.rank:
            return ind1
        elif ind2.rank < ind1.rank:
            return ind2
        elif ind1.crowding_distance > ind2.crowding_distance:
            return ind1
        elif ind2.crowding_distance > ind1.crowding_distance:
            return ind2
        else:
            return random.choice([ind1, ind2])
    
    def select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        tournament_size = config.TOURNAMENT_SIZE
        tournament = random.sample(population, min(tournament_size, len(population)))

        if len(tournament) == 0:
            raise ValueError("Population is empty, cannot select parents")

        tournament.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        if len(tournament) == 1:
            return tournament[0], tournament[0]
        return tournament[0], tournament[1]
    
    def environmental_selection(self, combined: List[Individual], target_size: int) -> List[Individual]:
        fronts = self.fast_non_dominated_sort(combined)
        
        new_population = []
        front_idx = 0
        
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= target_size:
            self.crowding_distance_assignment(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1

        if len(new_population) < target_size and front_idx < len(fronts):
            remaining = target_size - len(new_population)
            self.crowding_distance_assignment(fronts[front_idx])
            fronts[front_idx].sort(key=lambda ind: ind.crowding_distance, reverse=True)
            new_population.extend(fronts[front_idx][:remaining])
        
        return new_population


mutation_operator = MutationOperator()
crossover_operator = CrossoverOperator()    
nsga2_selector = NSGA2Selector()