"""
AGILE NIDS - Threshold Optimization Tool

This script finds optimal decision thresholds to balance:
- Attack detection rate (recall)
- False positive rate (FPR)
- Overall F1 score

Supports:
- Global threshold optimization
- Per-attack-type threshold optimization
- Custom optimization targets (security-focused vs balanced)
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from comprehensive_testing import ComprehensiveNIDSTester


class ThresholdOptimizer:
    """Find optimal decision thresholds for AGILE NIDS."""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize optimizer.
        
        Args:
            experiment_dir: Path to trained models
        """
        self.experiment_dir = experiment_dir
        self.tester = ComprehensiveNIDSTester(experiment_dir, max_samples_per_dataset=50000)
        self.optimization_results = {}
        
    def load_dataset_predictions(self, dataset_name: str, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset and get decision scores.
        
        Args:
            dataset_name: Name of dataset
            file_path: Path to CSV file
            
        Returns:
            Tuple of (decision_scores, true_labels)
        """
        print(f"\nAnalyzing: {dataset_name}")
        
        # Load dataset
        features, labels = self.tester._load_dataset(file_path, dataset_name)
        
        if features is None or labels is None:
            return None, None
        
        # Get predictions with decision scores
        predictions = self.tester._predict_batch(features)
        
        return predictions['decision_scores'], labels
    
    def find_optimal_threshold(self, 
                               decision_scores: np.ndarray, 
                               true_labels: np.ndarray,
                               optimization_target: str = 'balanced') -> Dict:
        """
        Find optimal threshold for a dataset.
        
        Args:
            decision_scores: Decision scores from model
            true_labels: Ground truth labels
            optimization_target: 'balanced', 'security', 'precision'
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        # Try thresholds from 0.01 to 0.95
        thresholds = np.linspace(0.01, 0.95, 200)
        
        best_threshold = 0.5
        best_score = 0
        all_metrics = []
        
        for threshold in thresholds:
            predictions = (decision_scores >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            # Calculate FPR
            fp = np.sum((predictions == 1) & (true_labels == 0))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            all_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr
            })
            
            # Optimization scoring based on target
            if optimization_target == 'balanced':
                # Balance F1 score with FPR constraint (FPR < 10%)
                if fpr < 0.10:
                    score = f1
                else:
                    score = f1 * (1 - (fpr - 0.10))  # Penalize high FPR
                    
            elif optimization_target == 'security':
                # Maximize recall with FPR constraint (FPR < 15%)
                if fpr < 0.15:
                    score = recall + 0.3 * precision  # Prioritize recall
                else:
                    score = recall * (1 - (fpr - 0.15))
                    
            elif optimization_target == 'precision':
                # Maximize precision while maintaining recall > 50%
                if recall > 0.50:
                    score = precision + 0.2 * recall
                else:
                    score = precision * (recall / 0.50)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Get metrics at optimal threshold
        optimal_predictions = (decision_scores >= best_threshold).astype(int)
        
        return {
            'optimal_threshold': best_threshold,
            'precision': precision_score(true_labels, optimal_predictions, zero_division=0),
            'recall': recall_score(true_labels, optimal_predictions, zero_division=0),
            'f1_score': f1_score(true_labels, optimal_predictions, zero_division=0),
            'fpr': np.sum((optimal_predictions == 1) & (true_labels == 0)) / np.sum(true_labels == 0),
            'optimization_target': optimization_target,
            'all_metrics': all_metrics
        }
    
    def optimize_per_attack_type(self, optimization_target: str = 'balanced') -> Dict:
        """
        Find optimal thresholds for each attack type.
        
        Args:
            optimization_target: 'balanced', 'security', 'precision'
            
        Returns:
            Dictionary with optimal thresholds per attack type
        """
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION BY ATTACK TYPE")
        print("="*70)
        print(f"Optimization Target: {optimization_target.upper()}\n")
        
        results = {}
        
        # Optimize for each attack dataset
        attack_datasets = {
            'DDoS': 'GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'PortScan': 'GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'WebAttacks': 'GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        }
        
        for attack_name, file_path in attack_datasets.items():
            if not os.path.exists(file_path):
                print(f"  Skipping {attack_name}: file not found")
                continue
            
            decision_scores, labels = self.load_dataset_predictions(attack_name, file_path)
            
            if decision_scores is None:
                continue
            
            # Find optimal threshold
            result = self.find_optimal_threshold(decision_scores, labels, optimization_target)
            results[attack_name] = result
            
            # Print results
            print(f"\n{attack_name}:")
            print(f"  Optimal Threshold: {result['optimal_threshold']:.4f}")
            print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
            print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
            print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
            print(f"  FPR:       {result['fpr']:.4f} ({result['fpr']*100:.2f}%)")
        
        # Find global optimal threshold (average)
        if results:
            global_threshold = np.mean([r['optimal_threshold'] for r in results.values()])
            results['Global'] = {
                'optimal_threshold': global_threshold,
                'description': 'Average of per-attack-type thresholds'
            }
            
            print(f"\nGlobal Optimal Threshold (Average): {global_threshold:.4f}")
        
        self.optimization_results = results
        return results
    
    def visualize_optimization(self, output_dir: str = "threshold_optimization"):
        """
        Create visualizations of threshold optimization.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f"optimization_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in: {report_dir}")
        
        # Plot threshold vs metrics for each attack type
        for attack_name, result in self.optimization_results.items():
            if 'all_metrics' not in result:
                continue
            
            metrics_df = pd.DataFrame(result['all_metrics'])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{attack_name} - Threshold Analysis', fontsize=16, fontweight='bold')
            
            # Precision vs Threshold
            ax = axes[0, 0]
            ax.plot(metrics_df['threshold'], metrics_df['precision'], 'b-', linewidth=2)
            ax.axvline(result['optimal_threshold'], color='red', linestyle='--', 
                      label=f"Optimal: {result['optimal_threshold']:.4f}")
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Precision')
            ax.set_title('Precision vs Threshold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Recall vs Threshold
            ax = axes[0, 1]
            ax.plot(metrics_df['threshold'], metrics_df['recall'], 'g-', linewidth=2)
            ax.axvline(result['optimal_threshold'], color='red', linestyle='--',
                      label=f"Optimal: {result['optimal_threshold']:.4f}")
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Recall')
            ax.set_title('Recall vs Threshold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # F1 Score vs Threshold
            ax = axes[1, 0]
            ax.plot(metrics_df['threshold'], metrics_df['f1'], 'purple', linewidth=2)
            ax.axvline(result['optimal_threshold'], color='red', linestyle='--',
                      label=f"Optimal: {result['optimal_threshold']:.4f}")
            ax.set_xlabel('Threshold')
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score vs Threshold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # FPR vs Threshold
            ax = axes[1, 1]
            ax.plot(metrics_df['threshold'], metrics_df['fpr'], 'orange', linewidth=2)
            ax.axvline(result['optimal_threshold'], color='red', linestyle='--',
                      label=f"Optimal: {result['optimal_threshold']:.4f}")
            ax.axhline(0.10, color='gray', linestyle=':', label='FPR Target (10%)')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('False Positive Rate')
            ax.set_title('FPR vs Threshold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'{attack_name}_threshold_analysis.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Comparative bar chart
        attack_results = {k: v for k, v in self.optimization_results.items() if k != 'Global'}
        
        if attack_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            attack_names = list(attack_results.keys())
            thresholds = [attack_results[name]['optimal_threshold'] for name in attack_names]
            
            bars = ax.bar(attack_names, thresholds, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.7)
            
            # Add global threshold line if available
            if 'Global' in self.optimization_results:
                global_thresh = self.optimization_results['Global']['optimal_threshold']
                ax.axhline(global_thresh, color='red', linestyle='--', linewidth=2,
                          label=f'Global Optimal: {global_thresh:.4f}')
            
            ax.set_ylabel('Optimal Threshold', fontsize=12)
            ax.set_title('Optimal Thresholds by Attack Type', fontsize=14, fontweight='bold')
            ax.set_ylim([0, max(thresholds) * 1.2])
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            # Add value labels
            for bar, val in zip(bars, thresholds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'threshold_comparison.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Visualizations saved\n")
        return report_dir
    
    def save_optimized_config(self, output_dir: str):
        """
        Save optimized thresholds to deployment config.
        
        Args:
            output_dir: Directory to save config
        """
        if not self.optimization_results:
            print("No optimization results to save")
            return
        
        config = {
            'optimization_timestamp': datetime.now().isoformat(),
            'optimization_target': self.optimization_results[list(self.optimization_results.keys())[0]].get('optimization_target', 'balanced'),
            'thresholds': {}
        }
        
        for attack_name, result in self.optimization_results.items():
            if 'optimal_threshold' in result:
                config['thresholds'][attack_name] = {
                    'threshold': float(result['optimal_threshold']),
                    'precision': float(result.get('precision', 0)),
                    'recall': float(result.get('recall', 0)),
                    'f1_score': float(result.get('f1_score', 0)),
                    'fpr': float(result.get('fpr', 0))
                }
        
        config_path = os.path.join(output_dir, 'optimized_thresholds.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Saved optimized config: {config_path}")


def main():
    """Main execution."""
    print("="*70)
    print("AGILE NIDS - THRESHOLD OPTIMIZATION TOOL")
    print("="*70)
    print("\nThis tool finds optimal decision thresholds to improve:")
    print("  - Attack detection rate (recall)")
    print("  - False positive rate (FPR)")
    print("  - Overall performance (F1 score)")
    print("="*70 + "\n")
    
    # Find latest experiment
    experiments = [d for d in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', d))]
    if not experiments:
        print("ERROR: No trained models found")
        return
    
    latest_experiment = max(experiments, key=lambda x: os.path.getctime(os.path.join('experiments', x)))
    experiment_dir = os.path.join('experiments', latest_experiment)
    
    print(f"Using models from: {latest_experiment}\n")
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(experiment_dir)
    
    # Optimization targets to try
    targets = {
        'balanced': 'Balanced (F1 score with FPR < 10%)',
        'security': 'Security-focused (maximize recall with FPR < 15%)',
        'precision': 'Precision-focused (minimize false alarms)'
    }
    
    print("\nOptimization Targets:")
    for i, (key, desc) in enumerate(targets.items(), 1):
        print(f"  {i}. {desc}")
    
    # Use balanced by default (or let user choose)
    selected_target = 'balanced'
    
    print(f"\nRunning optimization with target: {targets[selected_target]}")
    
    # Optimize thresholds
    results = optimizer.optimize_per_attack_type(optimization_target=selected_target)
    
    # Generate visualizations
    report_dir = optimizer.visualize_optimization()
    
    # Save optimized config
    optimizer.save_optimized_config(report_dir)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {report_dir}")
    print("\nNext steps:")
    print("  1. Review threshold_comparison.png to see optimal thresholds")
    print("  2. Check individual attack analysis plots")
    print("  3. Use optimized_thresholds.json in your deployment")
    print("  4. Re-run comprehensive_testing.py with new thresholds")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
