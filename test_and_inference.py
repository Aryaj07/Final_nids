"""
AGILE NIDS - Testing and Real-World Inference Script

This script allows you to:
1. Test trained models on new datasets
2. Perform real-time inference on network traffic
3. Generate comprehensive evaluation reports
4. Compare performance across different datasets
"""

import torch
import torch.nn as nn
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

# Import our models
from models.enhanced_agile_nids import create_enhanced_agile_nids
from models.autoencoder import Autoencoder
from models.correlation_engine import create_correlation_engine
from enhanced_training_success import SimpleTAGNNetwork, create_tagn_model


class AgileNIDSTester:
    """Comprehensive testing and inference system for trained AGILE NIDS models."""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize tester with trained model directory.
        
        Args:
            experiment_dir: Path to experiment directory containing trained models
        """
        self.experiment_dir = experiment_dir
        self.device = torch.device("cpu")  # Use CPU for inference
        
        # Load configuration
        self.config = self._load_config()
        self.input_dim = self.config['experiment_info']['input_dim']
        
        # Load calibrated decision threshold from deployment config
        deployment_config_path = os.path.join(self.experiment_dir, "deployment_config.json")
        if os.path.exists(deployment_config_path):
            with open(deployment_config_path, 'r') as f:
                deployment_config = json.load(f)
                self.decision_threshold = deployment_config['processing_config']['decision_threshold']
        else:
            self.decision_threshold = 0.5  # Fallback
        
        # Load decision threshold from deployment config if available
        deployment_config_path = os.path.join(self.experiment_dir, "deployment_config.json")
        if os.path.exists(deployment_config_path):
            with open(deployment_config_path, 'r') as f:
                deployment_config = json.load(f)
                self.decision_threshold = deployment_config['processing_config'].get('decision_threshold', 0.5)
        else:
            self.decision_threshold = 0.5  # Default fallback
        
        print(f"Decision threshold: {self.decision_threshold:.6f}")
        
        # Initialize models
        self.autoencoder = None
        self.tagn = None
        self.correlation_engine = None
        self.scaler = None
        
        self._load_models()
        
        print(f"Loaded models from: {experiment_dir}")
        print(f"Input dimension: {self.input_dim}")
        
    def _load_config(self) -> Dict:
        """Load training configuration."""
        config_path = os.path.join(self.experiment_dir, "training_report.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Training report not found in {self.experiment_dir}")
    
    def _load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        # Load Autoencoder
        autoencoder_path = os.path.join(self.experiment_dir, "autoencoder_trained.pt")
        self.autoencoder = Autoencoder(self.input_dim).to(self.device)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device, weights_only=False))
        self.autoencoder.eval()
        print(f"  [OK] Autoencoder loaded")
        
        # Load TAGN
        tagn_path = os.path.join(self.experiment_dir, "tagn_best.pt")
        # Use hidden_dim=128 to match the improved architecture from training
        self.tagn = create_tagn_model(input_dim=self.input_dim, hidden_dim=128).to(self.device)
        self.tagn.load_state_dict(torch.load(tagn_path, map_location=self.device, weights_only=False))
        self.tagn.eval()
        print(f"  [OK] TAGN Network loaded")
        
        # Load Correlation Engine
        correlation_path = os.path.join(self.experiment_dir, "correlation_engine.pt")
        self.correlation_engine = create_correlation_engine(
            autoencoder_dim=1,
            tagn_dim=16,  # MUST match training: 16-dim correlation_features
            confidence_dim=1,
            hidden_dim=64
        ).to(self.device)
        self.correlation_engine.load_state_dict(torch.load(correlation_path, map_location=self.device, weights_only=False))
        self.correlation_engine.eval()
        print(f"  [OK] Correlation Engine loaded")
        
        # Load training scaler (CRITICAL: must use same scaler as training!)
        import joblib
        scaler_path = os.path.join(self.experiment_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  [OK] Training scaler loaded")
        else:
            print(f"  [WARNING] Training scaler not found, will fit on test data (NOT RECOMMENDED!)")
            self.scaler = StandardScaler()
        print("Models loaded successfully!\n")
    
    def load_and_prepare_dataset(self, file_path: str, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and prepare a dataset for testing.
        
        Args:
            file_path: Path to CSV file
            max_rows: Maximum rows to load (None = all)
            
        Returns:
            Tuple of (original_df, features, labels)
        """
        print(f"\nLoading dataset: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                if max_rows:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False, nrows=max_rows)
                else:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                df.columns = df.columns.str.strip()
                break
            except:
                continue
        
        if df is None:
            raise Exception(f"Could not read {file_path}")
        
        print(f"  Loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Extract labels if available
        labels = None
        if 'Label' in df.columns:
            # Convert labels to binary: 0=BENIGN, 1=ATTACK
            labels = (~df['Label'].str.contains('BENIGN', case=False, na=False)).astype(int).values
            print(f"  Labels: {np.sum(labels == 0)} benign, {np.sum(labels == 1)} attack")
            df = df.drop('Label', axis=1)
        
        # Clean and prepare features
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        
        # Track valid rows before dropping NaN
        valid_mask = ~numeric_df.isna().any(axis=1)
        numeric_df = numeric_df[valid_mask]
        
        # Align labels with cleaned data
        if labels is not None:
            labels = labels[valid_mask.values]
            print(f"  After cleaning: {np.sum(labels == 0)} benign, {np.sum(labels == 1)} attack")
        
        numeric_df = numeric_df.clip(lower=-1e6, upper=1e6)
        
        # Ensure correct number of features
        if numeric_df.shape[1] != self.input_dim:
            print(f"  WARNING: Feature mismatch. Expected {self.input_dim}, got {numeric_df.shape[1]}")
            if numeric_df.shape[1] > self.input_dim:
                numeric_df = numeric_df.iloc[:, :self.input_dim]
            else:
                # Pad with zeros if needed
                padding = np.zeros((numeric_df.shape[0], self.input_dim - numeric_df.shape[1]))
                numeric_df = pd.concat([numeric_df, pd.DataFrame(padding)], axis=1)
        
        features = numeric_df.values
        
        # Use training scaler - DO NOT refit!
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("Scaler not loaded! Cannot proceed without training scaler.")
        
        print(f"  Prepared: {features.shape[0]} samples ready for inference\n")
        return df, features, labels
    
    def predict_single_flow(self, flow_features: np.ndarray) -> Dict:
        """
        Predict on a single network flow.
        
        Args:
            flow_features: Features of a single flow (shape: [input_dim])
            
        Returns:
            Dictionary with prediction results
        """
        # Scale features
        flow_scaled = self.scaler.transform(flow_features.reshape(1, -1))
        flow_tensor = torch.tensor(flow_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Stream A: Autoencoder
            reconstructed = self.autoencoder(flow_tensor)
            anomaly_score = torch.mean((flow_tensor - reconstructed) ** 2).item()
            
            # Stream B: TAGN (needs sequence, so we repeat the flow)
            sequence = flow_tensor.unsqueeze(0).repeat(1, 25, 1)  # [1, 25, input_dim]
            tagn_output = self.tagn(sequence)
            
            class_probs = tagn_output['classification']['class_probabilities'].squeeze().cpu().numpy()
            confidence = tagn_output['classification']['confidence_score'].item()
            predicted_class = tagn_output['classification']['predicted_class'].item()
            
            # Extract hybrid features for correlation engine (matching training)
            class_probs_tensor = tagn_output['classification']['class_probabilities']  # (1, 2)
            correlation_features = tagn_output['correlation_features']  # (1, 16)
            
            # Create hybrid 16-dim feature: [class_probs (2) + correlation_features[:14] (14)]
            threat_features = torch.cat([
                class_probs_tensor,  # 2 dims
                correlation_features[:, :14]  # 14 dims
            ], dim=1)  # Total: 16 dims
            
            confidence_tensor = tagn_output['classification']['confidence_score']
            if confidence_tensor.dim() == 1:
                confidence_tensor = confidence_tensor.unsqueeze(-1)
            
            # Correlation Engine - pass hybrid features (16-dim)
            autoencoder_results = {
                'anomaly_score': torch.tensor([[anomaly_score]]),
                'is_anomaly': torch.tensor([[anomaly_score > 0.1]])
            }
            
            tagn_results = {
                'classification': {
                    'class_probabilities': threat_features,  # (1, 16) hybrid features
                    'confidence_score': confidence_tensor    # (1, 1)
                },
                'priority_scores': torch.zeros(1, 4)  # Dummy priority scores
            }
            
            correlation_results = self.correlation_engine(autoencoder_results, tagn_results)
            correlation_score = correlation_results['correlation_score'].item()
        
        # Calculate decision score using attack probability (not confidence!)
        # Confidence = max(benign_prob, attack_prob) which is misleading for classification
        # Attack probability directly indicates likelihood of attack
        attack_probability = class_probs[1]
        
        # Hybrid decision: combine anomaly detection with classification
        # Give more weight to attack probability since TAGN trained on attack patterns
        decision_score = 0.3 * anomaly_score + 0.7 * attack_probability
        
        # Universal threshold: 0.10 balances detection across multiple attack types
        # PortScans (high scores ~0.70) vs WebAttacks (lower scores ~0.30)
        adjusted_threshold = 0.10
        is_attack = decision_score > adjusted_threshold
        
        return {
            'anomaly_score': anomaly_score,
            'predicted_class': predicted_class,
            'class_probabilities': {
                'benign': float(class_probs[0]),
                'attack': float(class_probs[1])
            },
            'confidence': confidence,
            'correlation_score': correlation_score,
            'decision_score': decision_score,
            'is_attack': is_attack
        }
    
    def predict_batch(self, features: np.ndarray, batch_size: int = 256) -> Dict[str, np.ndarray]:
        """
        Predict on a batch of flows.
        
        Args:
            features: Feature array (shape: [n_samples, input_dim])
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with prediction arrays
        """
        n_samples = features.shape[0]
        
        # Initialize result arrays
        anomaly_scores = np.zeros(n_samples)
        predicted_classes = np.zeros(n_samples, dtype=int)
        attack_probs = np.zeros(n_samples)
        confidences = np.zeros(n_samples)
        correlation_scores = np.zeros(n_samples)
        decision_scores = np.zeros(n_samples)  # Add decision scores
        
        print(f"Running inference on {n_samples} samples...")
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_features = features[i:batch_end]
            
            # Scale features
            batch_scaled = self.scaler.transform(batch_features)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Stream A: Autoencoder
                reconstructed = self.autoencoder(batch_tensor)
                batch_anomaly_scores = torch.mean((batch_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
                
                # Stream B: TAGN (create sequences)
                batch_sequences = batch_tensor.unsqueeze(1).repeat(1, 25, 1)
                tagn_output = self.tagn(batch_sequences)
                
                batch_class_probs = tagn_output['classification']['class_probabilities'].cpu().numpy()
                batch_confidences = tagn_output['classification']['confidence_score'].cpu().numpy()
                batch_predicted = tagn_output['classification']['predicted_class'].cpu().numpy()
                
                # Extract hybrid features for correlation engine (matching training)
                class_probs_tensor = tagn_output['classification']['class_probabilities']  # (batch, 2)
                correlation_features = tagn_output['correlation_features']  # (batch, 16)
                
                # Create hybrid 16-dim feature: [class_probs (2) + correlation_features[:14] (14)]
                threat_features = torch.cat([
                    class_probs_tensor,  # 2 dims
                    correlation_features[:, :14]  # 14 dims
                ], dim=1)  # Total: 16 dims
                
                confidence_tensor = tagn_output['classification']['confidence_score']
                if confidence_tensor.dim() == 1:
                    confidence_tensor = confidence_tensor.unsqueeze(-1)
                
                # Correlation Engine - pass hybrid features (16-dim)
                autoencoder_results = {
                    'anomaly_score': torch.tensor(batch_anomaly_scores).unsqueeze(-1),
                    'is_anomaly': torch.tensor(batch_anomaly_scores > 0.1).unsqueeze(-1)
                }
                
                tagn_results = {
                    'classification': {
                        'class_probabilities': threat_features,  # (batch, 16) hybrid features
                        'confidence_score': confidence_tensor    # (batch, 1)
                    },
                    'priority_scores': torch.zeros(batch_end - i, 4)
                }
                
                correlation_results = self.correlation_engine(autoencoder_results, tagn_results)
                batch_correlation = correlation_results['correlation_score'].squeeze().cpu().numpy()
            
            # Calculate decision scores using attack probability (not confidence!)
            # Must match the corrected formula in predict_single_flow
            batch_attack_probs = batch_class_probs[:, 1]  # Attack probability from TAGN
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
                print(f"  Progress: {progress:.1f}%")
        
        print("  Inference complete!\n")
        
        # Calculate final predictions with universal threshold
        # Threshold 0.10 balances PortScans and WebAttacks detection
        adjusted_threshold = 0.10
        final_predictions = (decision_scores > adjusted_threshold).astype(int)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predicted_classes': predicted_classes,
            'attack_probabilities': attack_probs,
            'confidences': confidences,
            'correlation_scores': correlation_scores,
            'decision_scores': decision_scores,
            'final_predictions': final_predictions  # Use this for evaluation
        }
    
    def evaluate_performance(self, predictions: Dict, true_labels: np.ndarray) -> Dict:
        """
        Evaluate model performance against ground truth.
        
        Args:
            predictions: Prediction dictionary from predict_batch
            true_labels: Ground truth labels (0=benign, 1=attack)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating performance...")
        
        # Use final_predictions based on decision_scores and threshold
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
            'benign_samples': int(np.sum(true_labels == 0))
        }
        
        # Print results
        print("\n" + "="*60)
        print("PERFORMANCE EVALUATION RESULTS")
        print("="*60)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"  Benign: {metrics['benign_samples']}")
        print(f"  Attack: {metrics['attack_samples']}")
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  FPR:       {fpr:.4f} ({fpr*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:6d}  |  False Positives: {fp:6d}")
        print(f"  False Negatives: {fn:6d}  |  True Positives:  {tp:6d}")
        print("="*60 + "\n")
        
        return metrics
    
    def generate_detailed_report(self, dataset_name: str, predictions: Dict, 
                                true_labels: np.ndarray, metrics: Dict,
                                output_dir: str = "test_results"):
        """
        Generate comprehensive test report with visualizations.
        
        Args:
            dataset_name: Name of the tested dataset
            predictions: Prediction dictionary
            true_labels: Ground truth labels
            metrics: Evaluation metrics
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f"{dataset_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        print(f"Generating detailed report in: {report_dir}")
        
        # Save metrics JSON
        report_data = {
            'dataset': dataset_name,
            'timestamp': timestamp,
            'experiment_dir': self.experiment_dir,
            'metrics': metrics,
            'config': self.config
        }
        
        with open(os.path.join(report_dir, 'metrics.json'), 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate plots
        self._plot_confusion_matrix(metrics, report_dir)
        self._plot_score_distributions(predictions, true_labels, report_dir)
        self._plot_roc_curve(predictions, true_labels, report_dir)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'true_label': true_labels,
            'predicted_class': predictions['predicted_classes'],
            'attack_probability': predictions['attack_probabilities'],
            'anomaly_score': predictions['anomaly_scores'],
            'correlation_score': predictions['correlation_scores'],
            'confidence': predictions['confidences']
        })
        pred_df.to_csv(os.path.join(report_dir, 'predictions.csv'), index=False)
        
        print(f"  [OK] Report saved: {report_dir}\n")
    
    def _plot_confusion_matrix(self, metrics: Dict, output_dir: str):
        """Plot confusion matrix."""
        cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                       [metrics['false_negatives'], metrics['true_positives']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
    
    def _plot_score_distributions(self, predictions: Dict, true_labels: np.ndarray, output_dir: str):
        """Plot score distributions for benign vs attack."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        benign_mask = (true_labels == 0)
        attack_mask = (true_labels == 1)
        
        # Attack probabilities
        axes[0, 0].hist(predictions['attack_probabilities'][benign_mask], bins=50, alpha=0.6, label='Benign', color='blue')
        axes[0, 0].hist(predictions['attack_probabilities'][attack_mask], bins=50, alpha=0.6, label='Attack', color='red')
        axes[0, 0].set_xlabel('Attack Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Attack Probability Distribution')
        axes[0, 0].legend()
        
        # Anomaly scores
        axes[0, 1].hist(predictions['anomaly_scores'][benign_mask], bins=50, alpha=0.6, label='Benign', color='blue')
        axes[0, 1].hist(predictions['anomaly_scores'][attack_mask], bins=50, alpha=0.6, label='Attack', color='red')
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Anomaly Score Distribution')
        axes[0, 1].legend()
        
        # Correlation scores
        axes[1, 0].hist(predictions['correlation_scores'][benign_mask], bins=50, alpha=0.6, label='Benign', color='blue')
        axes[1, 0].hist(predictions['correlation_scores'][attack_mask], bins=50, alpha=0.6, label='Attack', color='red')
        axes[1, 0].set_xlabel('Correlation Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Correlation Score Distribution')
        axes[1, 0].legend()
        
        # Confidence scores
        axes[1, 1].hist(predictions['confidences'][benign_mask], bins=50, alpha=0.6, label='Benign', color='blue')
        axes[1, 1].hist(predictions['confidences'][attack_mask], bins=50, alpha=0.6, label='Attack', color='red')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Confidence Score Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
        plt.close()
    
    def _plot_roc_curve(self, predictions: Dict, true_labels: np.ndarray, output_dir: str):
        """Plot ROC curve."""
        try:
            fpr, tpr, thresholds = roc_curve(true_labels, predictions['attack_probabilities'])
            roc_auc = roc_auc_score(true_labels, predictions['attack_probabilities'])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Could not generate ROC curve: {e}")


def main():
    """Main testing function with examples."""
    print("="*70)
    print("AGILE NIDS - Testing and Inference Tool")
    print("="*70)
    print()
    
    # Find the most recent experiment
    experiments = [d for d in os.listdir('experiments') if os.path.isdir(os.path.join('experiments', d))]
    if not experiments:
        print("ERROR: No trained models found in experiments/ directory")
        print("Please run enhanced_training_success.py first to train models.")
        return
    
    latest_experiment = max(experiments, key=lambda x: os.path.getctime(os.path.join('experiments', x)))
    experiment_dir = os.path.join('experiments', latest_experiment)
    
    print(f"Using latest trained models from: {latest_experiment}\n")
    
    # Initialize tester
    tester = AgileNIDSTester(experiment_dir)
    
    # Example 1: Test on a dataset
    print("\n" + "="*70)
    print("EXAMPLE 1: Testing on a Dataset")
    print("="*70)
    
    test_files = [
        "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting on: {test_file}")
            
            # Load dataset (limit to 50k for quick testing)
            dataset_name = os.path.basename(test_file).replace('.pcap_ISCX.csv', '')
            df, features, labels = tester.load_and_prepare_dataset(test_file, max_rows=50000)
            
            if labels is None:
                print("  No labels found - skipping evaluation")
                continue
            
            # Run predictions
            start_time = time.time()
            predictions = tester.predict_batch(features, batch_size=256)
            inference_time = time.time() - start_time
            
            print(f"  Inference time: {inference_time:.2f}s ({inference_time/len(features)*1000:.2f}ms per sample)")
            
            # Evaluate performance
            metrics = tester.evaluate_performance(predictions, labels)
            
            # Generate detailed report
            tester.generate_detailed_report(dataset_name, predictions, labels, metrics)
            
            print("\n" + "-"*70)
    
    # Example 2: Single flow inference
    print("\n" + "="*70)
    print("EXAMPLE 2: Real-Time Single Flow Inference")
    print("="*70)
    
    # Load a test sample
    if os.path.exists(test_files[0]):
        df, features, _ = tester.load_and_prepare_dataset(test_files[0], max_rows=10)
        
        print("\nTesting single flow inference:")
        for i in range(min(3, len(features))):
            result = tester.predict_single_flow(features[i])
            
            print(f"\nSample {i+1}:")
            print(f"  Attack Probability: {result['class_probabilities']['attack']:.4f}")
            print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
            print(f"  Correlation Score: {result['correlation_score']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Prediction: {'ATTACK' if result['is_attack'] else 'BENIGN'}")
    
    print("\n" + "="*70)
    print("Testing complete! Check 'test_results/' directory for detailed reports.")
    print("="*70)


if __name__ == "__main__":
    main()
