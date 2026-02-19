import random
import os
import pickle
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from configuration.config import config
from core.encoding import Individual
from core.search_space import search_space
from search.mutation import mutation_operator, nsga2_selector, crossover_operator
from engine.evaluator import ntk_evaluator, final_evaluator
from utils.logger import logger


class NSGA2NAS:
    def __init__(self):
        self.population_size = config.POPULATION_SIZE
        self.max_gen = config.MAX_GEN
        self.mutation_prob = config.MUTATION_PROB
        
        self.population: List[Individual] = []
        self.history: List[Individual] = []
        self.gen_stats: List[dict] = []    
        
        self.start_time = time.time()
        self.search_time = 0.0
        self.short_train_time = 0.0
        self.full_train_time = 0.0
        self.time_stats: dict = {}
        self.current_gen = 0
    
        self.archive: dict[tuple,Individual] = {}  
        self._seen_keys: set = set()
        
        self._log_search_space_info()
    
    def _log_search_space_info(self):
        logger.info((
            f"=== NSGA-II NAS Configuration ===\n"
            f"Random Seed: {config.RANDOM_SEED}\n"
            f"Nodes per Cell: {config.NUM_NODES}\n"
            f"Edges per Node: {config.EDGES_PER_NODE}\n"
            f"Operations: {len(config.OPERATIONS)}\n"
            f"Initial Channels: {config.INIT_CHANNELS}\n"
            f"Cells per Stage: {config.CELLS_PER_STAGE}\n"
            f"Stages: {config.NUM_STAGES}\n"
            f"Total Cells: {config.NUM_STAGES * (config.CELLS_PER_STAGE + 1) - 1}\n"
            f"Population Size: {self.population_size}\n"
            f"Max Generations: {self.max_gen}\n"
            f"Mutation Prob: {self.mutation_prob}\n"
            f"Objectives: [NTK Condition Number (min), -K logdet (min)]\n"
        ))
    
    def _get_genotype_key(self, ind: Individual) -> tuple:
        genotype = ind.get_genotype()
        return (tuple(genotype['normal']), tuple(genotype['reduce']))
    
    def _update_archive(self, individuals: List[Individual]):
        for ind in individuals:
            if ind.ntk_score is None or ind.k_score is None:
                continue
            key = self._get_genotype_key(ind)
            self._seen_keys.add(key)
            if key not in self.archive:
                self.archive[key] = ind.copy()
            else:
                existing = self.archive[key]
                if ind.dominates(existing):
                    self.archive[key] = ind.copy()
    
    def initialize_population(self):
        logger.info(f"Initializing population of {self.population_size} individuals...")
        
        self.population = []
        max_attempts = self.population_size * 10  
        attempts = 0
        while len(self.population) < self.population_size and attempts < max_attempts:
            ind = search_space.sample_individual()
            key = self._get_genotype_key(ind)
            if key not in self._seen_keys:
                self._seen_keys.add(key)
                self.population.append(ind)
            attempts += 1
        
        n_dup_skipped = attempts - len(self.population)
        if n_dup_skipped > 0:
            logger.info(f"  Init sampling: skipped {n_dup_skipped} duplicates")
        
        ntk_evaluator.evaluate_population(self.population)
        self._update_archive(self.population)
        
        fronts = nsga2_selector.fast_non_dominated_sort(self.population)
        for front in fronts:
            nsga2_selector.crowding_distance_assignment(front)
        
        self.history.extend(self.population)
        
        self._record_generation_stats(0)
        logger.info(f"Init done. Pareto front size: {len(fronts[0])}, Archive: {len(self.archive)}, Unique seen: {len(self._seen_keys)}")
    
    def _generate_offspring_population(self) -> List[Individual]:
        offspring = []
        offspring_keys = set() 
        n_batch_dup = 0
        n_global_dup = 0
        max_attempts = self.population_size * 50 
        attempts = 0
        
        while len(offspring) < self.population_size and attempts < max_attempts:
            attempts += 1
            parent1, parent2 = nsga2_selector.select_parents(self.population)
            
            child = crossover_operator.crossover(parent1, parent2)
            
            if random.random() < self.mutation_prob:
                child = mutation_operator.mutate(child)
            
            key = self._get_genotype_key(child)
            
            if key in offspring_keys:
                n_batch_dup += 1
                continue
            
            if key in self._seen_keys:
                n_global_dup += 1
                continue
            
            offspring_keys.add(key)
            offspring.append(child)
        
        if n_batch_dup > 0 or n_global_dup > 0:
            logger.info(f"  Offspring gen: {len(offspring)} novel | "
                       f"{n_batch_dup} batch-dups | "
                       f"{n_global_dup} global-dups | "
                       f"{attempts} attempts")
        
        return offspring
    
    def step(self):
        gen = self.current_gen + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {gen}/{self.max_gen}")
        logger.info(f"{'='*60}")
        
        gen_start = time.time()
        offspring = self._generate_offspring_population()
        
        ntk_evaluator.evaluate_population(offspring)
        
        self._update_archive(offspring)
        
        combined = self.population + offspring
        
        self.population = nsga2_selector.environmental_selection(combined, self.population_size)
        
        self.history.extend(offspring)
        self.current_gen = gen
        gen_time = time.time() - gen_start
        
        self._record_generation_stats(gen, gen_time)
        self._plot_pareto_front()
        
        # 定期保存
        if gen % 5 == 0:
            self._save_search_history()
        if gen % 10 == 0:
            self._save_checkpoint()
    
    def _record_generation_stats(self, gen: int, gen_time: float = 0.0):
        fronts = nsga2_selector.fast_non_dominated_sort(self.population)
        for front in fronts:
            nsga2_selector.crowding_distance_assignment(front)
        
        pareto_front = fronts[0]
        
        ntk_scores = [ind.ntk_score for ind in self.population 
                      if ind.ntk_score is not None and ind.ntk_score < config.NTK_FAIL_SCORE]
        k_scores = [ind.k_score for ind in self.population 
                        if ind.k_score is not None and ind.k_score > config.K_FAIL_SCORE]
        
        stats = {
            'generation': gen,
            'gen_time': gen_time,
            'population_size': len(self.population),
            'num_fronts': len(fronts),
            'pareto_front_size': len(pareto_front),
            'archive_size': len(self.archive),
            'ntk_best': min(ntk_scores) if ntk_scores else config.NTK_FAIL_SCORE,
            'ntk_mean': sum(ntk_scores) / len(ntk_scores) if ntk_scores else config.NTK_FAIL_SCORE,
            'k_best': max(k_scores) if k_scores else config.K_FAIL_SCORE,
            'k_mean': sum(k_scores) / len(k_scores) if k_scores else config.K_FAIL_SCORE,
        }
        self.gen_stats.append(stats)
        
        logger.info(
            f"Gen {gen}: "
            f"Fronts={len(fronts)} | "
            f"ParetoSize={len(pareto_front)} | "
            f"Archive={len(self.archive)} | "
            f"NTK best={stats['ntk_best']:.4f} mean={stats['ntk_mean']:.4f} | "
            f"K best={stats['k_best']:.4f} mean={stats['k_mean']:.4f} | "
            f"Time={gen_time:.1f}s"
        )
    
    def run_search(self):
        logger.info(f"Starting NSGA-II Search for {self.max_gen} generations...")
        search_start_time = time.time()
        
        if not self.population:
            self.initialize_population()
        
        while self.current_gen < self.max_gen:
            self.step()
                
        self.search_time = time.time() - search_start_time
        logger.info(f"\nSearch completed in {self._format_time(self.search_time)}")
        self._save_checkpoint()
        self._save_search_history()
        self._plot_pareto_front()
        self._plot_search_curves()
    
    def _get_global_top_candidates(self, top_n: int) -> List[Individual]:
        archive_list = list(self.archive.values())
        if not archive_list:
            return []
            
        valid = [ind for ind in archive_list
                 if ind.ntk_score is not None and ind.k_score is not None
                 and ind.ntk_score < config.NTK_FAIL_SCORE
                 and ind.k_score > config.K_FAIL_SCORE]
                 
        if not valid:
            logger.info("No valid candidates found in archive.")
            return []
            
        n = len(valid)
        
        sorted_by_ntk = sorted(range(n), key=lambda i: valid[i].ntk_score)
        ntk_rank = [0] * n
        for rank, idx in enumerate(sorted_by_ntk):
            ntk_rank[idx] = rank
            
        sorted_by_k = sorted(range(n), key=lambda i: -valid[i].k_score)
        k_rank = [0] * n
        for rank, idx in enumerate(sorted_by_k):
            k_rank[idx] = rank
            
        combined = [(ntk_rank[i] + k_rank[i], i) for i in range(n)]
        combined.sort(key=lambda x: x[0])
        
        result = []
        for score, idx in combined[:top_n]:
            ind = valid[idx]
            ind.combined_rank_score = float(score)
            result.append(ind)
            
        return result
    
    def run_screening_and_training(self):
        logger.info("\n" + "=" * 60)
        logger.info("Starting Screening and Training Phase (NSGA-II Pareto)")
        logger.info("=" * 60)

        top_n1 = self._get_global_top_candidates(config.HISTORY_TOP_N1)
        logger.info(f"Selected Top {len(top_n1)} candidates from Global Archive by combined NTK+K rank.")
        
        for i, ind in enumerate(top_n1):
            logger.info(f"  Candidate {i+1}: ID={ind.id}, CombinedRank={ind.combined_rank_score:.0f}, "
                       f"NTK={ind.ntk_score:.4f}, K={ind.k_score:.4f}")
        
        logger.info(f"\nStarting Short Training ({config.SHORT_TRAIN_EPOCHS} epochs) for Top {len(top_n1)}...")
        short_train_start = time.time()
        
        short_results = []
        for i, ind in enumerate(top_n1):
            logger.info(f"Short Train [{i+1}/{len(top_n1)}] ID: {ind.id}")
            acc, _ = final_evaluator.evaluate_individual(ind, full_train=False)
            ind.accuracy = acc
            short_results.append(ind)
        
        self.short_train_time = time.time() - short_train_start
        logger.info(f"Short Training completed. Time: {self._format_time(self.short_train_time)}")
        
        short_results.sort(key=lambda x: x.accuracy if x.accuracy else float('-inf'), reverse=True)
        top_n2 = short_results[:config.HISTORY_TOP_N2]
        logger.info(f"\nSelected Top {len(top_n2)} candidates based on Short Training Accuracy.")
        
        logger.info(f"\nStarting Full Training ({config.FULL_TRAIN_EPOCHS} epochs) for Top {len(top_n2)}...")
        full_train_start = time.time()
        
        final_results = []
        best_final_ind = None
        best_final_acc = 0.0
        
        for i, ind in enumerate(top_n2):
            logger.info(f"Full Train [{i+1}/{len(top_n2)}] ID: {ind.id}")
            acc, result = final_evaluator.evaluate_individual(ind)
            
            logger.info(f"Individual {ind.id} Final Accuracy: {acc:.2f}%")
            
            if acc > best_final_acc:
                best_final_acc = acc
                best_final_ind = ind
                
            final_results.append(result)
        
        self.full_train_time = time.time() - full_train_start
        logger.info(f"Full Training completed. Time: {self._format_time(self.full_train_time)}")
        
        self._save_time_stats()
            
        logger.info(f"\nBest Final Model: ID={best_final_ind.id}, Acc={best_final_acc:.2f}%")
        return best_final_ind
    
    def _save_checkpoint(self):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        filepath = os.path.join(config.CHECKPOINT_DIR, f'nsga2_checkpoint_gen{self.current_gen}.pkl')
        
        population_data = [ind.to_dict() for ind in self.population]
        history_data = [ind.to_dict() for ind in self.history]
        archive_data = {str(k): v.to_dict() for k, v in self.archive.items()}
        
        checkpoint = {
            'population': population_data,
            'history': history_data,
            'archive': archive_data,
            'seen_keys': list(self._seen_keys),
            'gen_stats': self.gen_stats,
            'current_gen': self.current_gen,
            'search_time': self.search_time,
            'short_train_time': self.short_train_time,
            'full_train_time': self.full_train_time,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Checkpoint saved to {filepath} (archive: {len(self.archive)} unique archs)")
        
    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        population_data = checkpoint.get('population', [])
        history_data = checkpoint.get('history', [])
        
        if population_data:
            self.population = [Individual.from_dict(d) for d in population_data]
            self.history = [Individual.from_dict(d) for d in history_data]
        
        archive_data = checkpoint.get('archive', {})
        for k_str, v_data in archive_data.items():
            ind = Individual.from_dict(v_data)
            key = self._get_genotype_key(ind)
            self.archive[key] = ind
            self._seen_keys.add(key)
        
        saved_keys = checkpoint.get('seen_keys', [])
        for k in saved_keys:
            self._seen_keys.add(tuple(k) if isinstance(k, list) else k)
        
        self.gen_stats = checkpoint.get('gen_stats', [])
        self.current_gen = checkpoint.get('current_gen', 0)
        self.search_time = checkpoint.get('search_time', 0.0)
        self.short_train_time = checkpoint.get('short_train_time', 0.0)
        self.full_train_time = checkpoint.get('full_train_time', 0.0)
        
        if self.history:
            max_id = max(ind.id for ind in self.history)
            Individual.update_id_counter(max_id)
            
        logger.info(f"Checkpoint loaded from {filepath} (gen={self.current_gen})")
    
    def _save_search_history(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
    
        stats_path = os.path.join(config.LOG_DIR, 'nsga2_gen_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.gen_stats, f, indent=2, ensure_ascii=False, default=str)
        
        pop_data = []
        for ind in self.population:
            genotype = ind.get_genotype()
            pop_data.append({
                'id': ind.id,
                'rank': ind.rank,
                'crowding_distance': ind.crowding_distance,
                'ntk_score': ind.ntk_score,
                'k_score': ind.k_score,
                'objectives': ind.objectives,
                'param_count': ind.param_count,
                'genotype': {
                    'normal': genotype.get('normal', []),
                    'reduce': genotype.get('reduce', [])
                }
            })
        
        pop_path = os.path.join(config.LOG_DIR, 'nsga2_population.json')
        with open(pop_path, 'w', encoding='utf-8') as f:
            json.dump(pop_data, f, indent=2, ensure_ascii=False, default=str)
        
        archive_data = [ind.to_dict() for ind in self.archive.values()]
        archive_path = os.path.join(config.LOG_DIR, 'nsga2_archive.json')
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Search history saved to {config.LOG_DIR} (archive: {len(self.archive)} unique archs)")
    
    def _plot_pareto_front(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        output_path = os.path.join(config.LOG_DIR, f'pareto_front_gen{self.current_gen}.png')
        
        fronts = nsga2_selector.fast_non_dominated_sort(self.population)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_fronts_to_plot = min(len(fronts), 5)
        colors = plt.cm.viridis(np.linspace(0, 1, num_fronts_to_plot))
        
        for i, front in enumerate(fronts[:num_fronts_to_plot]):
            valid_inds = [ind for ind in front 
                         if ind.ntk_score is not None and ind.k_score is not None
                         and ind.ntk_score < config.NTK_FAIL_SCORE 
                         and ind.k_score > config.K_FAIL_SCORE]
            ntk_vals = [ind.ntk_score for ind in valid_inds]
            k_vals = [ind.k_score for ind in valid_inds]
            
            if ntk_vals and k_vals:
                ax.scatter(ntk_vals, k_vals, 
                          c=[colors[i]], s=50, alpha=0.7,
                          label=f'Front {i} (n={len(valid_inds)})',
                          edgecolors='black', linewidth=0.5)
                
                if i == 0 and len(ntk_vals) > 1:
                    sorted_pairs = sorted(zip(ntk_vals, k_vals))
                    sorted_ntk = [p[0] for p in sorted_pairs]
                    sorted_k = [p[1] for p in sorted_pairs]
                    ax.plot(sorted_ntk, sorted_k, 
                           c=colors[i], linewidth=1.5, alpha=0.6, linestyle='-')
        
        ax.set_xlabel('NTK Condition Number (lower is better)', fontsize=12)
        ax.set_ylabel('K logdet (higher is better)', fontsize=12)
        ax.set_title(f'NSGA-II Pareto Fronts (Gen {self.current_gen})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Pareto front plot saved to {output_path}")
    
    def _plot_search_curves(self):
        if not self.gen_stats:
            return
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        output_path = os.path.join(config.LOG_DIR, 'nsga2_search_curves.png')
        
        gens = [s['generation'] for s in self.gen_stats]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'NSGA-II Search Progress ({self.max_gen} Generations)', fontsize=14, fontweight='bold')
        
        # 1. NTK 条件数趋势
        ax1 = axes[0, 0]
        ntk_best = [s['ntk_best'] for s in self.gen_stats]
        ntk_mean = [s['ntk_mean'] for s in self.gen_stats]
        ax1.plot(gens, ntk_best, 'b-', linewidth=2, label='Best NTK')
        ax1.plot(gens, ntk_mean, 'b--', alpha=0.5, label='Mean NTK')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('NTK Condition Number')
        ax1.set_title('NTK Score Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. K logdet 趋势
        ax2 = axes[0, 1]
        k_best = [s['k_best'] for s in self.gen_stats]
        k_mean = [s['k_mean'] for s in self.gen_stats]
        ax2.plot(gens, k_best, 'r-', linewidth=2, label='Best K')
        ax2.plot(gens, k_mean, 'r--', alpha=0.5, label='Mean K')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('K logdet')
        ax2.set_title('K Score Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Pareto 前沿大小和前沿数量
        ax3 = axes[1, 0]
        pareto_sizes = [s['pareto_front_size'] for s in self.gen_stats]
        num_fronts = [s['num_fronts'] for s in self.gen_stats]
        ax3.plot(gens, pareto_sizes, 'g-', linewidth=2, label='Pareto Front Size')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(gens, num_fronts, 'orange', linewidth=2, label='Num Fronts')
        ax3_twin.set_ylabel('Number of Fronts', color='orange')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Pareto Front Size', color='green')
        ax3.set_title('Population Diversity')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 每代耗时
        ax4 = axes[1, 1]
        gen_times = [s.get('gen_time', 0) for s in self.gen_stats]
        ax4.plot(gens, gen_times, 'purple', linewidth=1.5)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Generation Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Search curves saved to {output_path}")
    
    # ==================== 工具方法 ====================
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}min ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f}h ({minutes:.0f}min)"
    
    def _save_time_stats(self):
        total_time = self.search_time + self.short_train_time + self.full_train_time
        
        self.time_stats = {
            'search_phase': {
                'time_seconds': self.search_time,
                'time_formatted': self._format_time(self.search_time),
                'description': f'NSGA-II搜索阶段 ({self.max_gen} 代, 种群大小 {self.population_size})'
            },
            'short_training_phase': {
                'time_seconds': self.short_train_time,
                'time_formatted': self._format_time(self.short_train_time),
                'description': f'短轮次训练阶段 (Top {config.HISTORY_TOP_N1} 个模型, {config.SHORT_TRAIN_EPOCHS} epochs)'
            },
            'full_training_phase': {
                'time_seconds': self.full_train_time,
                'time_formatted': self._format_time(self.full_train_time),
                'description': f'完整训练阶段 (Top {config.HISTORY_TOP_N2} 个模型, {config.FULL_TRAIN_EPOCHS} epochs)'
            },
            'total': {
                'time_seconds': total_time,
                'time_formatted': self._format_time(total_time),
                'description': '总耗时'
            }
        }
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        filepath = os.path.join(config.LOG_DIR, 'time_stats.json')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.time_stats, f, indent=2, ensure_ascii=False)
        
        logger.info("=" * 60)
        logger.info("时间统计总结")
        logger.info("=" * 60)
        logger.info(f"NSGA-II搜索:     {self._format_time(self.search_time)}")
        logger.info(f"短轮次训练:       {self._format_time(self.short_train_time)}")
        logger.info(f"完整训练:         {self._format_time(self.full_train_time)}")
        logger.info("-" * 60)
        logger.info(f"总耗时:           {self._format_time(total_time)}")
        logger.info("=" * 60)
        logger.info(f"Time stats saved to {filepath}")
