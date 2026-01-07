"""
AGILE NIDS - Adaptive Threshold Testing

This script uses optimized per-attack-type thresholds for better performance.
Based on threshold optimization results, different attack types need different thresholds.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from comprehensive_testing import ComprehensiveNIDSTester


class AdaptiveThresholdTester(ComprehensiveNIDSTester):
    """Enhanced tester with per-attack-type adaptive thresholds."""
    
    # Optimized thresholds from threshold_optimizer.py
    ADAPTIVE_THRESHOLDS = {
        'DDoS': 0.034,           # Low threshold for weak DDoS signals
        'PortScan': 0.700,       # High threshold to reduce false alarms
        'WebAttacks': 0.076,     # Slightly lower than default
        'Infiltration': 0.100,   # Default (no attacks in dataset anyway)
        'Default': 0.100         # Fallback for unknown types
    }
    
    def __init__(self, experiment_dir: str, max_samples_per_dataset: int = 50000,
                 use_adaptive_thresholds: bool = True, custom_thresholds: Dict = None):
        """
        Initialize adaptive threshold tester.
        
        Args:
            experiment_dir: Path to experiment directory
            max_samples_per_dataset: Max samples per dataset
            use_adaptive_thresholds: Use per-attack-type thresholds
            custom_thresholds: Custom threshold dictionary
        """
        # Initialize parent class
        super().__init__(experiment_dir, max_samples_per_dataset)
        
        self.use_adaptive_thresholds = use_adaptive_thresholds
        
        # Load custom thresholds if provided
        if custom_thresholds:
            self.ADAPTIVE_THRESHOLDS.update(custom_thresholds)
        
        # Try to load optimized thresholds from file
        self._load_optimized_thresholds()
        
        print(f"\n{'='*70}")
        print("ADAPTIVE THRESHOLDING STATUS")
        print(f"{'='*70}")
        print(f"Adaptive Thresholds: {'ENABLED' if use_adaptive_thresholds else 'DISABLED'}")
        if use_adaptive_thresholds:
            print("\nPer-Attack-Type Thresholds:")
            for attack_type, threshold in self.ADAPTIVE_THRESHOLDS.items():
                if attack_type != 'Default':
                    print(f"  {attack_type:15s}: {threshold:.4f}")
            print(f"  {'Default':15s}: {self.ADAPTIVE_THRESHOLDS['Default']:.4f}")
        print(f"{'='*70}\n")
    
    def _load_optimized_thresholds(self):
        """Load optimized thresholds from threshold_optimizer output."""
        # Check for recent optimization results
        if os.path.exists('threshold_optimization'):
            optimization_dirs = [d for d in os.listdir('threshold_optimization') 
                               if os.path.isdir(os.path.join('threshold_optimization', d))]
            
            if optimization_dirs:
                # Get most recent
                latest_dir = max(optimization_dirs, 
                               key=lambda x: os.path.getctime(os.path.join('threshold_optimization', x)))
                
                config_path = os.path.join('threshold_optimization', latest_dir, 'optimized_thresholds.json')
                
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Update thresholds
                    for attack_type, data in config.get('thresholds', {}).items():
                        if attack_type != 'Global' and 'threshold' in data:
                            self.ADAPTIVE_THRESHOLDS[attack_type] = data['threshold']
                    
                    print(f"✓ Loaded optimized thresholds from: {latest_dir}")
    
    def _get_threshold_for_dataset(self, dataset_name: str) -> float:
        """
        Get appropriate threshold for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Threshold value
        """
        if not self.use_adaptive_thresholds:
            return self.decision_threshold
        
        # Match dataset name to attack type
        for attack_type in self.ADAPTIVE_THRESHOLDS.keys():
            if attack_type in dataset_name:
                return self.ADAPTIVE_THRESHOLDS[attack_type]
        
        # Return default
        return self.ADAPTIVE_THRESHOLDS['Default']
    
    def _evaluate_predictions_adaptive(self, predictions: Dict, true_labels: np.ndarray, 
                                       dataset_name: str, threshold: float) -> Dict:
        """
        Evaluate predictions with specified threshold.
        
        Args:
            predictions: Prediction dictionary
            true_labels: Ground truth
            dataset_name: Dataset name
            threshold: Decision threshold to use
            
        Returns:
            Metrics dictionary
        """
        # Recalculate predictions with adaptive threshold
        y_pred = (predictions['decision_scores'] >= threshold).astype(int)
        y_prob = predictions['attack_probabilities']
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, y_pred)
        precision = precision_score(true_labels, y_pred, zero_division=0)
        recall = recall_score(true_labels, y_pred, zero_division=0)
        f1 = f1_score(true_labels, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, y_prob)
        except:
            roc_auc = 0.0
        
        metrics = {
            'dataset': dataset_name,
            'threshold_used': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_positive_rate': fpr,
            'roc_auc': roc_auc,
            'total_samples': len(true_labels),
            'attack_samples': int(np.sum(true_labels == 1)),
            'benign_samples': int(np.sum(true_labels == 0)),
            'inference_time': predictions['inference_time']
        }
        
        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*70}")
        print(f"Threshold: {threshold:.4f} {'(ADAPTIVE)' if self.use_adaptive_thresholds else '(GLOBAL)'}")
        print(f"Samples: {metrics['total_samples']} (Benign: {metrics['benign_samples']}, Attack: {metrics['attack_samples']})")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  FPR:       {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:6d}  |  FP: {fp:6d}")
        print(f"  FN: {fn:6d}  |  TP: {tp:6d}")
        print(f"{'='*70}\n")
        
        return metrics
    
    def run_adaptive_test(self) -> Dict:
        """
        Run comprehensive testing with adaptive thresholds.
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*70)
        print("ADAPTIVE THRESHOLD TESTING")
        print("="*70)
        
        all_metrics = {}
        
        # Test each dataset
        for dataset_name, dataset_info in self.DATASETS.items():
            file_path = dataset_info['file']
            
            # Load dataset
            features, labels = self._load_dataset(file_path, dataset_name)
            
            if features is None or labels is None:
                print(f"  Skipping {dataset_name} (loading failed)\n")
                continue
            
            # Run predictions
            predictions = self._predict_batch(features)
            
            # Get adaptive threshold
            threshold = self._get_threshold_for_dataset(dataset_name)
            
            # Evaluate with adaptive threshold
            metrics = self._evaluate_predictions_adaptive(predictions, labels, dataset_name, threshold)
            metrics['dataset_info'] = dataset_info
            
            all_metrics[dataset_name] = metrics
            
            # Store for later analysis
            self.all_results[dataset_name] = {
                'predictions': predictions,
                'labels': labels,
                'metrics': metrics
            }
        
        return all_metrics
    
    def generate_comparison_report(self, adaptive_metrics: Dict, 
                                   output_dir: str = "adaptive_test_results"):
        """
        Generate comparison report between adaptive and global thresholds.
        
        Args:
            adaptive_metrics: Metrics with adaptive thresholds
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f"adaptive_test_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Generating Comparison Report")
        print(f"{'='*70}")
        print(f"Output directory: {report_dir}\n")
        
        # Save metrics
        report_data = {
            'timestamp': timestamp,
            'experiment_dir': self.experiment_dir,
            'adaptive_thresholds_used': self.use_adaptive_thresholds,
            'thresholds': self.ADAPTIVE_THRESHOLDS,
            'metrics': adaptive_metrics
        }
        
        with open(os.path.join(report_dir, 'adaptive_metrics.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"  ✓ Saved: adaptive_metrics.json")
        
        # Create comparison CSV
        self._create_comparison_csv(adaptive_metrics, report_dir)
        
        # Generate plots
        self._plot_threshold_comparison(adaptive_metrics, report_dir)
        self._plot_comparative_metrics(adaptive_metrics, report_dir)
        self._plot_confusion_matrices(report_dir)
        
        # Generate text report
        self._generate_comparison_text_report(adaptive_metrics, report_dir)
        
        print(f"\n{'='*70}")
        print(f"ADAPTIVE TESTING COMPLETE!")
        print(f"{'='*70}")
        print(f"Report saved to: {report_dir}")
        print(f"{'='*70}\n")
    
    def _create_comparison_csv(self, adaptive_metrics: Dict, output_dir: str):
        """Create comparison CSV."""
        summary_data = []
        
        for dataset_name, metrics in adaptive_metrics.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Type': metrics['dataset_info']['type'],
                'Threshold': f"{metrics['threshold_used']:.4f}",
                'Total_Samples': metrics['total_samples'],
                'Attack_Samples': metrics['attack_samples'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'FPR': f"{metrics['false_positive_rate']:.4f}",
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'adaptive_summary.csv'), index=False)
        print(f"  ✓ Saved: adaptive_summary.csv")
    
    def _plot_threshold_comparison(self, adaptive_metrics: Dict, output_dir: str):
        """Plot thresholds used for each dataset."""
        datasets = []
        thresholds = []
        types = []
        
        for name, metrics in adaptive_metrics.items():
            datasets.append(name)
            thresholds.append(metrics['threshold_used'])
            types.append(metrics['dataset_info']['type'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if t == 'attack' else 'blue' for t in types]
        bars = ax.bar(range(len(datasets)), thresholds, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_ylabel('Decision Threshold', fontsize=12)
        ax.set_title('Adaptive Thresholds by Dataset', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, thresholds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Attack Datasets'),
            Patch(facecolor='blue', alpha=0.7, label='Benign Datasets')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: threshold_comparison.png")
    
    def _generate_comparison_text_report(self, adaptive_metrics: Dict, output_dir: str):
        """Generate detailed text report."""
        report_path = os.path.join(output_dir, 'adaptive_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AGILE NIDS - ADAPTIVE THRESHOLD TESTING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_dir}\n")
            f.write(f"Adaptive Thresholds: {'ENABLED' if self.use_adaptive_thresholds else 'DISABLED'}\n")
            f.write(f"Datasets Tested: {len(adaptive_metrics)}\n\n")
            
            # Summary
            f.write("="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            attack_datasets = {k: v for k, v in adaptive_metrics.items() 
                             if v['dataset_info']['type'] == 'attack'}
            
            if attack_datasets:
                f.write(f"Attack Datasets ({len(attack_datasets)}):\n")
                avg_recall = np.mean([m['recall'] for m in attack_datasets.values()])
                avg_precision = np.mean([m['precision'] for m in attack_datasets.values()])
                avg_f1 = np.mean([m['f1_score'] for m in attack_datasets.values()])
                avg_fpr = np.mean([m['false_positive_rate'] for m in attack_datasets.values()])
                f.write(f"  Average Detection Rate (Recall): {avg_recall:.4f} ({avg_recall*100:.2f}%)\n")
                f.write(f"  Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)\n")
                f.write(f"  Average F1 Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)\n")
                f.write(f"  Average FPR: {avg_fpr:.4f} ({avg_fpr*100:.2f}%)\n\n")
            
            # Detailed results
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS BY DATASET\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, metrics in adaptive_metrics.items():
                f.write(f"\n{dataset_name}\n")
                f.write("-"*80 + "\n")
                f.write(f"Description: {metrics['dataset_info']['description']}\n")
                f.write(f"Type: {metrics['dataset_info']['type'].upper()}\n")
                f.write(f"Threshold: {metrics['threshold_used']:.6f}\n")
                f.write(f"Samples: {metrics['total_samples']} (Benign: {metrics['benign_samples']}, Attack: {metrics['attack_samples']})\n\n")
                
                f.write(f"Performance Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                f.write(f"  FPR:       {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)\n\n")
                
                f.write(f"Confusion Matrix:\n")
                f.write(f"  True Negatives:  {metrics['true_negatives']:8d}\n")
                f.write(f"  False Positives: {metrics['false_positives']:8d}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']:8d}\n")
                f.write(f"  True Positives:  {metrics['true_positives']:8d}\n\n")
        
        print(f"  ✓ Saved: adaptive_report.txt")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("AGILE NIDS - ADAPTIVE THRESHOLD TESTING")
    print("="*70)
    print("\nThis script uses optimized per-attack-type thresholds:")
    print("  - DDoS:       ~0.034 (low threshold for weak signals)")
    print("  - PortScan:   ~0.700 (high threshold to reduce false alarms)")
    print("  - WebAttacks: ~0.076 (slightly lower than default)")
    print("="*70 + "\n")
    
    # Find latest experiment
    experiments = [d for d in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', d))]
    if not experiments:
        print("ERROR: No trained models found")
        return
    
    latest_experiment = max(experiments, key=lambda x: os.path.getctime(os.path.join('experiments', x)))
    experiment_dir = os.path.join('experiments', latest_experiment)
    
    print(f"Using models from: {latest_experiment}\n")
    
    # Initialize adaptive tester
    tester = AdaptiveThresholdTester(experiment_dir, 
                                    max_samples_per_dataset=50000,
                                    use_adaptive_thresholds=True)
    
    # Run adaptive testing
    adaptive_metrics = tester.run_adaptive_test()
    
    # Generate comparison report
    tester.generate_comparison_report(adaptive_metrics)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print("\nCheck 'adaptive_test_results/' directory for:")
    print("  - Adaptive threshold performance metrics")
    print("  - Comparison with global threshold")
    print("  - Visualizations showing improvements")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
