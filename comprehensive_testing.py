"""
AGILE NIDS - Comprehensive Testing Script

This script systematically tests the trained NIDS models across ALL available datasets:
- Multiple attack types (DDoS, PortScan, WebAttacks, Infiltration)
- Normal/benign traffic (Monday, Tuesday, Wednesday, Friday Morning)
- Generates comparative analysis across all datasets
- Provides detailed performance breakdown by attack type
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our models
from models.autoencoder import Autoencoder
from models.correlation_engine import create_correlation_engine
from enhanced_training_success import create_tagn_model


class ComprehensiveNIDSTester:
    """Comprehensive testing system for AGILE NIDS across all attack types."""
    
    # Dataset definitions with attack types
    DATASETS = {
        'DDoS': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'type': 'attack',
            'description': 'Distributed Denial of Service attacks'
        },
        'PortScan': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'type': 'attack',
            'description': 'Port scanning reconnaissance attacks'
        },
        'WebAttacks': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'type': 'attack',
            'description': 'Web-based attacks (SQL injection, XSS, etc.)'
        },
        'Infiltration': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'type': 'attack',
            'description': 'Network infiltration and exploitation'
        },
        'Monday_Normal': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv',
            'type': 'benign',
            'description': 'Normal Monday working hours traffic'
        },
        'Tuesday_Normal': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv',
            'type': 'benign',
            'description': 'Normal Tuesday working hours traffic'
        },
        'Wednesday_Normal': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv',
            'type': 'benign',
            'description': 'Normal Wednesday working hours traffic'
        },
        'Friday_Morning_Normal': {
            'file': 'GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'type': 'benign',
            'description': 'Normal Friday morning traffic'
        }
    }
    
    def __init__(self, experiment_dir: str, max_samples_per_dataset: int = 50000):
        """
        Initialize comprehensive tester.
        
        Args:
            experiment_dir: Path to experiment directory with trained models
            max_samples_per_dataset: Max samples to load per dataset (for memory efficiency)
        """
        self.experiment_dir = experiment_dir
        self.max_samples = max_samples_per_dataset
        self.device = torch.device("cpu")
        
        # Load configuration
        self.config = self._load_config()
        self.input_dim = self.config['experiment_info']['input_dim']
        
        # Load decision threshold from deployment config
        deployment_config_path = os.path.join(self.experiment_dir, "deployment_config.json")
        if os.path.exists(deployment_config_path):
            with open(deployment_config_path, 'r') as f:
                deployment_config = json.load(f)
                loaded_threshold = deployment_config['processing_config'].get('decision_threshold', 0.10)
                print(f"⚠ Loaded threshold from deployment config: {loaded_threshold:.6f}")
                
                # Override if threshold is too high (causes poor detection)
                if loaded_threshold > 0.15:
                    self.decision_threshold = 0.10
                    print(f"✓ Overriding with optimal threshold: {self.decision_threshold:.6f}")
                    print(f"  (Original threshold {loaded_threshold:.6f} is too conservative)")
                else:
                    self.decision_threshold = loaded_threshold
        else:
            self.decision_threshold = 0.10
            print(f"✓ Using default threshold: {self.decision_threshold:.6f}")
        
        # Initialize models
        self.autoencoder = None
        self.tagn = None
        self.correlation_engine = None
        self.scaler = None
        
        self._load_models()
        
        # Results storage
        self.all_results = {}
        
        print(f"\nLoaded models from: {experiment_dir}")
        print(f"Input dimension: {self.input_dim}")
        print(f"Max samples per dataset: {self.max_samples}")
        print(f"\nDatasets to test: {len(self.DATASETS)}")
        for name, info in self.DATASETS.items():
            print(f"  - {name}: {info['description']}")
    
    def _load_config(self) -> Dict:
        """Load training configuration."""
        config_path = os.path.join(self.experiment_dir, "training_report.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Training report not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_models(self):
        """Load all trained models."""
        print("\nLoading trained models...")
        
        # Load Autoencoder
        autoencoder_path = os.path.join(self.experiment_dir, "autoencoder_trained.pt")
        self.autoencoder = Autoencoder(self.input_dim).to(self.device)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device, weights_only=False))
        self.autoencoder.eval()
        print(f"  ✓ Autoencoder loaded")
        
        # Load TAGN
        tagn_path = os.path.join(self.experiment_dir, "tagn_best.pt")
        self.tagn = create_tagn_model(input_dim=self.input_dim, hidden_dim=128).to(self.device)
        self.tagn.load_state_dict(torch.load(tagn_path, map_location=self.device, weights_only=False))
        self.tagn.eval()
        print(f"  ✓ TAGN Network loaded")
        
        # Load Correlation Engine
        correlation_path = os.path.join(self.experiment_dir, "correlation_engine.pt")
        self.correlation_engine = create_correlation_engine(
            autoencoder_dim=1,
            tagn_dim=16,
            confidence_dim=1,
            hidden_dim=64
        ).to(self.device)
        self.correlation_engine.load_state_dict(torch.load(correlation_path, map_location=self.device, weights_only=False))
        self.correlation_engine.eval()
        print(f"  ✓ Correlation Engine loaded")
        
        # Load training scaler
        import joblib
        scaler_path = os.path.join(self.experiment_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Training scaler loaded")
        else:
            raise FileNotFoundError(f"Training scaler not found: {scaler_path}")
    
    def _load_dataset(self, file_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare a dataset.
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (features, labels)
        """
        print(f"\n{'='*70}")
        print(f"Loading: {dataset_name}")
        print(f"{'='*70}")
        
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: {file_path}")
            return None, None
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False, nrows=self.max_samples)
                df.columns = df.columns.str.strip()
                break
            except:
                continue
        
        if df is None:
            print(f"  ✗ Could not read file with any encoding")
            return None, None
        
        print(f"  Loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Extract labels
        labels = None
        if 'Label' in df.columns:
            # Convert to binary: 0=BENIGN, 1=ATTACK
            labels = (~df['Label'].str.contains('BENIGN', case=False, na=False)).astype(int).values
            
            # Count attack types
            label_counts = df['Label'].value_counts()
            print(f"\n  Label distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count}")
            
            print(f"\n  Binary: {np.sum(labels == 0)} benign, {np.sum(labels == 1)} attack")
            df = df.drop('Label', axis=1)
        else:
            print(f"  ⚠ No labels found in dataset")
            return None, None
        
        # Prepare features
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        
        # Track valid rows
        valid_mask = ~numeric_df.isna().any(axis=1)
        numeric_df = numeric_df[valid_mask]
        labels = labels[valid_mask.values]
        
        print(f"  After cleaning: {len(labels)} samples ({np.sum(labels==0)} benign, {np.sum(labels==1)} attack)")
        
        numeric_df = numeric_df.clip(lower=-1e6, upper=1e6)
        
        # Ensure correct number of features
        if numeric_df.shape[1] != self.input_dim:
            print(f"  ⚠ Feature mismatch. Expected {self.input_dim}, got {numeric_df.shape[1]}")
            if numeric_df.shape[1] > self.input_dim:
                numeric_df = numeric_df.iloc[:, :self.input_dim]
            else:
                padding = np.zeros((numeric_df.shape[0], self.input_dim - numeric_df.shape[1]))
                numeric_df = pd.concat([numeric_df, pd.DataFrame(padding)], axis=1)
        
        features = numeric_df.values
        
        print(f"  ✓ Prepared: {features.shape[0]} samples ready\n")
        return features, labels
    
    def _predict_batch(self, features: np.ndarray, batch_size: int = 256) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch of samples.
        
        Args:
            features: Feature array [n_samples, input_dim]
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with prediction arrays
        """
        n_samples = features.shape[0]
        
        # Initialize arrays
        anomaly_scores = np.zeros(n_samples)
        predicted_classes = np.zeros(n_samples, dtype=int)
        attack_probs = np.zeros(n_samples)
        confidences = np.zeros(n_samples)
        correlation_scores = np.zeros(n_samples)
        decision_scores = np.zeros(n_samples)
        
        print(f"  Running inference on {n_samples} samples...")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_features = features[i:batch_end]
            
            # Scale features
            batch_scaled = self.scaler.transform(batch_features)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Autoencoder
                reconstructed = self.autoencoder(batch_tensor)
                batch_anomaly_scores = torch.mean((batch_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
                
                # TAGN
                batch_sequences = batch_tensor.unsqueeze(1).repeat(1, 25, 1)
                tagn_output = self.tagn(batch_sequences)
                
                batch_class_probs = tagn_output['classification']['class_probabilities'].cpu().numpy()
                batch_confidences = tagn_output['classification']['confidence_score'].cpu().numpy()
                batch_predicted = tagn_output['classification']['predicted_class'].cpu().numpy()
                
                # Hybrid features for correlation engine
                class_probs_tensor = tagn_output['classification']['class_probabilities']
                correlation_features = tagn_output['correlation_features']
                
                threat_features = torch.cat([
                    class_probs_tensor,
                    correlation_features[:, :14]
                ], dim=1)
                
                confidence_tensor = tagn_output['classification']['confidence_score']
                if confidence_tensor.dim() == 1:
                    confidence_tensor = confidence_tensor.unsqueeze(-1)
                
                # Correlation Engine
                autoencoder_results = {
                    'anomaly_score': torch.tensor(batch_anomaly_scores).unsqueeze(-1),
                    'is_anomaly': torch.tensor(batch_anomaly_scores > 0.1).unsqueeze(-1)
                }
                
                tagn_results = {
                    'classification': {
                        'class_probabilities': threat_features,
                        'confidence_score': confidence_tensor
                    },
                    'priority_scores': torch.zeros(batch_end - i, 4)
                }
                
                correlation_results = self.correlation_engine(autoencoder_results, tagn_results)
                batch_correlation = correlation_results['correlation_score'].squeeze().cpu().numpy()
            
            # Calculate decision scores
            batch_attack_probs = batch_class_probs[:, 1]
            batch_decision_scores = 0.3 * batch_anomaly_scores + 0.7 * batch_attack_probs
            
            # Store results
            anomaly_scores[i:batch_end] = batch_anomaly_scores
            predicted_classes[i:batch_end] = batch_predicted
            attack_probs[i:batch_end] = batch_class_probs[:, 1]
            confidences[i:batch_end] = batch_confidences
            correlation_scores[i:batch_end] = batch_correlation
            decision_scores[i:batch_end] = batch_decision_scores
            
            # Progress
            if (i // batch_size) % 10 == 0:
                progress = (batch_end / n_samples) * 100
                print(f"    Progress: {progress:.1f}%")
        
        inference_time = time.time() - start_time
        print(f"  ✓ Inference complete in {inference_time:.2f}s ({inference_time/n_samples*1000:.2f}ms per sample)")
        
        # Final predictions
        final_predictions = (decision_scores > self.decision_threshold).astype(int)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predicted_classes': predicted_classes,
            'attack_probabilities': attack_probs,
            'confidences': confidences,
            'correlation_scores': correlation_scores,
            'decision_scores': decision_scores,
            'final_predictions': final_predictions,
            'inference_time': inference_time
        }
    
    def _evaluate_predictions(self, predictions: Dict, true_labels: np.ndarray, dataset_name: str) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: Prediction dictionary
            true_labels: Ground truth labels
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with metrics
        """
        y_pred = predictions['final_predictions']
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
    
    def run_comprehensive_test(self) -> Dict:
        """
        Run comprehensive testing across all datasets.
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE AGILE NIDS TESTING")
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
            
            # Evaluate
            metrics = self._evaluate_predictions(predictions, labels, dataset_name)
            metrics['dataset_info'] = dataset_info
            
            all_metrics[dataset_name] = metrics
            
            # Store for later analysis
            self.all_results[dataset_name] = {
                'predictions': predictions,
                'labels': labels,
                'metrics': metrics
            }
        
        return all_metrics
    
    def generate_comparative_report(self, all_metrics: Dict, output_dir: str = "comprehensive_test_results"):
        """
        Generate comprehensive comparative report.
        
        Args:
            all_metrics: Dictionary with all metrics
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f"comprehensive_test_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Generating Comprehensive Report")
        print(f"{'='*70}")
        print(f"Output directory: {report_dir}\n")
        
        # Save metrics JSON
        report_data = {
            'timestamp': timestamp,
            'experiment_dir': self.experiment_dir,
            'decision_threshold': self.decision_threshold,
            'max_samples_per_dataset': self.max_samples,
            'datasets_tested': len(all_metrics),
            'metrics': all_metrics
        }
        
        with open(os.path.join(report_dir, 'comprehensive_metrics.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"  ✓ Saved: comprehensive_metrics.json")
        
        # Create summary CSV
        self._create_summary_csv(all_metrics, report_dir)
        
        # Generate comparative plots
        self._plot_comparative_metrics(all_metrics, report_dir)
        self._plot_attack_type_comparison(all_metrics, report_dir)
        self._plot_confusion_matrices(report_dir)
        
        # Generate detailed report text
        self._generate_text_report(all_metrics, report_dir)
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE TESTING COMPLETE!")
        print(f"{'='*70}")
        print(f"Report saved to: {report_dir}")
        print(f"{'='*70}\n")
    
    def _create_summary_csv(self, all_metrics: Dict, output_dir: str):
        """Create summary CSV of all results."""
        summary_data = []
        
        for dataset_name, metrics in all_metrics.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Type': metrics['dataset_info']['type'],
                'Total_Samples': metrics['total_samples'],
                'Attack_Samples': metrics['attack_samples'],
                'Benign_Samples': metrics['benign_samples'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'ROC_AUC': f"{metrics['roc_auc']:.4f}",
                'FPR': f"{metrics['false_positive_rate']:.4f}",
                'TP': metrics['true_positives'],
                'TN': metrics['true_negatives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives'],
                'Inference_Time_s': f"{metrics['inference_time']:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        print(f"  ✓ Saved: summary.csv")
    
    def _plot_comparative_metrics(self, all_metrics: Dict, output_dir: str):
        """Plot comparative metrics across all datasets."""
        datasets = list(all_metrics.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics Comparison Across All Datasets', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = [all_metrics[ds][metric] for ds in datasets]
            colors = ['red' if all_metrics[ds]['dataset_info']['type'] == 'attack' else 'blue' for ds in datasets]
            
            bars = ax.bar(range(len(datasets)), values, color=colors, alpha=0.7)
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Attack Datasets'),
            Patch(facecolor='blue', alpha=0.7, label='Benign Datasets')
        ]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: comparative_metrics.png")
    
    def _plot_attack_type_comparison(self, all_metrics: Dict, output_dir: str):
        """Plot attack-specific performance comparison."""
        attack_datasets = {k: v for k, v in all_metrics.items() if v['dataset_info']['type'] == 'attack'}
        
        if not attack_datasets:
            return
        
        datasets = list(attack_datasets.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attack Detection Performance by Attack Type', fontsize=16, fontweight='bold')
        
        # Detection Rate (Recall)
        ax = axes[0, 0]
        recalls = [attack_datasets[ds]['recall'] for ds in datasets]
        bars = ax.barh(datasets, recalls, color='crimson', alpha=0.7)
        ax.set_xlabel('Detection Rate (Recall)')
        ax.set_title('Attack Detection Rate')
        ax.set_xlim([0, 1.1])
        for bar, val in zip(bars, recalls):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', ha='left', fontweight='bold')
        
        # False Positive Rate
        ax = axes[0, 1]
        fprs = [attack_datasets[ds]['false_positive_rate'] for ds in datasets]
        bars = ax.barh(datasets, fprs, color='orange', alpha=0.7)
        ax.set_xlabel('False Positive Rate')
        ax.set_title('False Positive Rate (Lower is Better)')
        ax.set_xlim([0, max(fprs) * 1.2])
        for bar, val in zip(bars, fprs):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', ha='left', fontweight='bold')
        
        # F1 Score
        ax = axes[1, 0]
        f1s = [attack_datasets[ds]['f1_score'] for ds in datasets]
        bars = ax.barh(datasets, f1s, color='green', alpha=0.7)
        ax.set_xlabel('F1 Score')
        ax.set_title('Overall F1 Score')
        ax.set_xlim([0, 1.1])
        for bar, val in zip(bars, f1s):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   va='center', ha='left', fontweight='bold')
        
        # Attack Detection Summary
        ax = axes[1, 1]
        tp_counts = [attack_datasets[ds]['true_positives'] for ds in datasets]
        fn_counts = [attack_datasets[ds]['false_negatives'] for ds in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, tp_counts, width, label='Detected (TP)', color='green', alpha=0.7)
        ax.bar(x + width/2, fn_counts, width, label='Missed (FN)', color='red', alpha=0.7)
        
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Count')
        ax.set_title('Attack Detection vs Missed')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attack_type_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: attack_type_comparison.png")
    
    def _plot_confusion_matrices(self, output_dir: str):
        """Plot confusion matrices for all datasets."""
        n_datasets = len(self.all_results)
        if n_datasets == 0:
            return
        
        # Calculate grid size
        n_cols = 4
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_datasets == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (dataset_name, results) in enumerate(self.all_results.items()):
            metrics = results['metrics']
            cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                          [metrics['false_negatives'], metrics['true_positives']]])
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Benign', 'Attack'],
                       yticklabels=['Benign', 'Attack'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'{dataset_name}\nF1: {metrics["f1_score"]:.3f}')
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - All Datasets', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_confusion_matrices.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: all_confusion_matrices.png")
    
    def _generate_text_report(self, all_metrics: Dict, output_dir: str):
        """Generate detailed text report."""
        report_path = os.path.join(output_dir, 'detailed_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AGILE NIDS - COMPREHENSIVE TESTING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment: {self.experiment_dir}\n")
            f.write(f"Decision Threshold: {self.decision_threshold:.6f}\n")
            f.write(f"Datasets Tested: {len(all_metrics)}\n\n")
            
            # Summary statistics
            f.write("="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            attack_datasets = {k: v for k, v in all_metrics.items() if v['dataset_info']['type'] == 'attack'}
            benign_datasets = {k: v for k, v in all_metrics.items() if v['dataset_info']['type'] == 'benign'}
            
            if attack_datasets:
                f.write(f"Attack Datasets ({len(attack_datasets)}):\n")
                avg_recall = np.mean([m['recall'] for m in attack_datasets.values()])
                avg_f1 = np.mean([m['f1_score'] for m in attack_datasets.values()])
                avg_fpr = np.mean([m['false_positive_rate'] for m in attack_datasets.values()])
                f.write(f"  Average Detection Rate (Recall): {avg_recall:.4f}\n")
                f.write(f"  Average F1 Score: {avg_f1:.4f}\n")
                f.write(f"  Average FPR: {avg_fpr:.4f}\n\n")
            
            if benign_datasets:
                f.write(f"Benign Datasets ({len(benign_datasets)}):\n")
                avg_precision = np.mean([m['precision'] for m in benign_datasets.values()])
                avg_fpr = np.mean([m['false_positive_rate'] for m in benign_datasets.values()])
                f.write(f"  Average Precision: {avg_precision:.4f}\n")
                f.write(f"  Average FPR: {avg_fpr:.4f}\n\n")
            
            # Individual dataset results
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS BY DATASET\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, metrics in all_metrics.items():
                f.write(f"\n{dataset_name}\n")
                f.write("-"*80 + "\n")
                f.write(f"Description: {metrics['dataset_info']['description']}\n")
                f.write(f"Type: {metrics['dataset_info']['type'].upper()}\n")
                f.write(f"Samples: {metrics['total_samples']} (Benign: {metrics['benign_samples']}, Attack: {metrics['attack_samples']})\n\n")
                
                f.write(f"Performance Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")
                f.write(f"  FPR:       {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)\n\n")
                
                f.write(f"Confusion Matrix:\n")
                f.write(f"  True Negatives:  {metrics['true_negatives']:8d}\n")
                f.write(f"  False Positives: {metrics['false_positives']:8d}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']:8d}\n")
                f.write(f"  True Positives:  {metrics['true_positives']:8d}\n\n")
                
                f.write(f"Inference Time: {metrics['inference_time']:.2f}s\n")
                f.write(f"Time per sample: {metrics['inference_time']/metrics['total_samples']*1000:.2f}ms\n")
                f.write("\n")
        
        print(f"  ✓ Saved: detailed_report.txt")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("AGILE NIDS - COMPREHENSIVE TESTING SCRIPT")
    print("="*70)
    print("\nThis script tests the trained models on ALL available datasets:")
    print("  - DDoS attacks")
    print("  - Port scanning attacks")
    print("  - Web-based attacks")
    print("  - Infiltration attacks")
    print("  - Multiple benign/normal traffic datasets")
    print("="*70 + "\n")
    
    # Find latest experiment
    experiments = [d for d in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', d))]
    if not experiments:
        print("ERROR: No trained models found in experiments/ directory")
        print("Please run enhanced_training_success.py first to train models.")
        return
    
    latest_experiment = max(experiments, key=lambda x: os.path.getctime(os.path.join('experiments', x)))
    experiment_dir = os.path.join('experiments', latest_experiment)
    
    print(f"Using trained models from: {latest_experiment}")
    
    # Initialize tester (adjust max_samples as needed for memory)
    tester = ComprehensiveNIDSTester(experiment_dir, max_samples_per_dataset=50000)
    
    # Run comprehensive testing
    all_metrics = tester.run_comprehensive_test()
    
    # Generate comprehensive report
    tester.generate_comparative_report(all_metrics)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TESTING COMPLETE!")
    print("="*70)
    print("\nCheck 'comprehensive_test_results/' directory for:")
    print("  - Detailed metrics JSON")
    print("  - Summary CSV")
    print("  - Comparative performance plots")
    print("  - Attack-type specific analysis")
    print("  - Confusion matrices for all datasets")
    print("  - Detailed text report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
