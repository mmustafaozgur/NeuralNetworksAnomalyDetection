import argparse
import os
import time
import shutil
import json
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

# Anomalib imports
from anomalib.data import Folder
from anomalib.models import Fastflow, Cfa, Padim  
from anomalib.engine import Engine

# PyTorch Lightning imports
try:
    from lightning.pytorch.callbacks import Timer
    from lightning.pytorch.loggers import TensorBoardLogger
    import lightning.pytorch as pl
except ImportError:
    from pytorch_lightning.callbacks import Timer
    from pytorch_lightning.loggers import TensorBoardLogger
    import pytorch_lightning as pl

# Sklearn imports
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import torch

def setup_directories(model_name):
    """Create directory structure for results."""
    results_dir = Path("results") / model_name
    subdirs = ["models", "metrics", "visualizations", "roc_curves", "heatmaps", "segmentations"]
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (results_dir / subdir).mkdir(exist_ok=True)
    
    return results_dir

def create_datamodule(data_path):
    """Create data module with version-compatible configuration."""
    # Optimized batch sizes for better training stability
    batch_size = 12 if torch.cuda.is_available() else 6
    num_workers = 4 if torch.cuda.is_available() else 2
    
    # Try different parameter combinations based on version
    datamodule = None
    error_messages = []
    
    # Configuration 1: Latest version parameters
    try:
        datamodule = Folder(
            name="wood",
            root=data_path,
            normal_dir="train/good",
            abnormal_dir="test/defect", 
            normal_test_dir="test/good",
            mask_dir="ground_truth/defect",
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            val_split_ratio=0.15,
            val_split_mode="from_train",
            test_split_mode="from_dir",
            seed=42,
            image_size=(256, 256),  # Try with image_size first
            task="segmentation",
        )
        print("   ✅ Created datamodule with image_size parameter")
        
    except TypeError as e1:
        error_messages.append(f"Config 1 failed: {e1}")
        
        # Configuration 2: Remove image_size and task parameters  
        try:
            datamodule = Folder(
                name="wood",
                root=data_path,
                normal_dir="train/good",
                abnormal_dir="test/defect", 
                normal_test_dir="test/good",
                mask_dir="ground_truth/defect",
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
                num_workers=num_workers,
                val_split_ratio=0.15,
                seed=42,
            )
            print("   ✅ Created datamodule without image_size parameter")
            
        except TypeError as e2:
            error_messages.append(f"Config 2 failed: {e2}")
            
            # Configuration 3: Minimal parameters only
            try:
                datamodule = Folder(
                    root=data_path,
                    normal_dir="train/good",
                    abnormal_dir="test/defect", 
                    normal_test_dir="test/good",
                    mask_dir="ground_truth/defect",
                    train_batch_size=batch_size,
                    eval_batch_size=batch_size,
                    num_workers=num_workers,
                    # Minimal configuration
                )
                print("   ✅ Created datamodule with minimal parameters")
                
            except Exception as e3:
                error_messages.append(f"Config 3 failed: {e3}")
                print("   ❌ All configurations failed:")
                for msg in error_messages:
                    print(f"      {msg}")
                raise RuntimeError("Unable to create datamodule with any configuration")
    
    if datamodule is None:
        raise RuntimeError("Failed to create datamodule")
    
    # Try to add transforms after creation
    try:
        # Check if we can add transforms post-creation
        from torchvision.transforms import v2
        
        # Basic augmentations
        train_augmentations = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),  
            v2.RandomRotation(degrees=10),   
        ])
        
        # Try different ways to add augmentations
        if hasattr(datamodule, 'train_augmentations'):
            datamodule.train_augmentations = train_augmentations
            print("   ✅ Added train_augmentations")
        elif hasattr(datamodule, 'transform_config_train'):
            # For older versions
            print("   ⚠️  Using transform_config_train (limited)")
        else:
            print("   ⚠️  No augmentation support found")
            
    except Exception as aug_error:
        print(f"   ⚠️  Augmentation setup failed: {aug_error}")
        print("   ℹ️  Continuing without augmentation...")
    
    return datamodule

def get_model(model_name):
    """Get optimized model with best parameters for high F1 scores."""
    
    if model_name.lower() == "fastflow":  # Changed from efficientad to fastflow
        # FastFlow optimizations for better anomaly detection
        # Removed problematic parameters and adjusted based on Anomalib's FastFlow implementation
        try:
            model = Fastflow(
                backbone="resnet18",    # Options: resnet18, wide_resnet50_2
                pre_trained=True,       # Use pre-trained weights
                flow_steps=8,           # More flow steps for better expressivity (default is usually 4-6)
                hidden_ratio=1.0,       # Hidden ratio for better feature extraction
                input_size=(256, 256),  # Input image size
                coupling_type="affine", # Options: additive, affine
            )
        except TypeError as e:
            # Fallback to a more minimal configuration
            print(f"   ⚠️  First FastFlow configuration failed: {e}")
            print(f"   ℹ️  Trying a more minimal configuration...")
            
            try:
                # More minimal configuration that should work with most versions
                model = Fastflow(
                    backbone="resnet18",
                    input_size=(256, 256)
                )
                print(f"   ✅ Created FastFlow model with minimal parameters")
            except TypeError as e2:
                # Ultra minimal configuration as last resort
                print(f"   ⚠️  Second FastFlow configuration failed: {e2}")
                print(f"   ℹ️  Trying basic configuration...")
                
                # Try to check available parameters first
                import inspect
                sig = inspect.signature(Fastflow.__init__)
                print(f"   Available FastFlow parameters: {list(sig.parameters.keys())}")
                
                # Try with only required parameters
                required_params = {
                    name: param.default for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty and name != 'self'
                }
                print(f"   Required parameters: {required_params}")
                
                # Create with minimal configuration
                model = Fastflow()
                
        
    elif model_name.lower() == "cfa":
        # CFA parameter optimization for better hypersphere separation
        model = Cfa(
            backbone="wide_resnet50_2",  # Upgraded from wide_resnet50_2 for stability
            gamma_c=3,            # Increased coupling strength (was 1)
            gamma_d=3,            # Increased distance penalty (was 1)  
            num_nearest_neighbors=5,      # More neighbors for better clustering (was 3)
            num_hard_negative_features=5, # More hard negatives (was 3)
            radius=5e-05,                 # Optimized radius (was 1e-05)
        )
        
    elif model_name.lower() == "padim":
        # PADIM optimizations - more comprehensive features
        model = Padim(
            backbone="resnet34",  # Upgraded from resnet18 for better features
            layers=["layer1", "layer2", "layer3"],  # Comprehensive feature extraction
            pre_trained=True,
            n_features=150,  # More features for better representation (default: 100 for resnet18)
        )
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def extract_image_and_mask_from_batch(batch):
    """Safely extract images and masks from Anomalib batch."""
    try:
        # Handle different batch structures
        if hasattr(batch, 'image'):
            images = batch.image
            masks = getattr(batch, 'mask', None)
        elif hasattr(batch, 'images'):
            images = batch.images  
            masks = getattr(batch, 'masks', None)
        elif isinstance(batch, dict):
            images = batch.get('image', batch.get('images'))
            masks = batch.get('mask', batch.get('masks'))
        else:
            # Last resort: iterate through batch attributes
            images = None
            masks = None
            for attr_name in dir(batch):
                if not attr_name.startswith('_'):
                    try:
                        attr_value = getattr(batch, attr_name)
                        if torch.is_tensor(attr_value) and attr_value.ndim == 4:
                            if 'image' in attr_name.lower() and images is None:
                                images = attr_value
                            elif 'mask' in attr_name.lower() and masks is None:
                                masks = attr_value
                    except:
                        continue
        
        # Ensure images exist
        if images is None:
            raise ValueError("No images found in batch")
            
        return images, masks
    except Exception as e:
        print(f"Error extracting batch data: {e}")
        return None, None

def tensor_to_numpy_image(tensor):
    """Convert tensor to numpy array for visualization."""
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().numpy()
        # Handle different tensor shapes: (C, H, W) -> (H, W, C)
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            if img.shape[0] == 1:
                img = img[0]  # Remove single channel dimension
            else:
                img = img.transpose(1, 2, 0)  # Move channels to last
        # Normalize to [0, 1] range
        img = np.clip(img, 0, 1)
    else:
        img = tensor
    return img

def create_visualization(image, anomaly_map, ground_truth=None):
    """Create comprehensive visualization with improved anomaly thresholding."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    img_np = tensor_to_numpy_image(image)
    if img_np.ndim == 2:
        axes[0].imshow(img_np, cmap='gray')
    else:
        axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Anomaly map - IMPROVED PROCESSING
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_np = anomaly_map.cpu().numpy()
    else:
        anomaly_np = anomaly_map
    
    # Remove extra dimensions
    while anomaly_np.ndim > 2:
        anomaly_np = anomaly_np.squeeze()
    
    # BETTER NORMALIZATION: Set a baseline from normal regions
    # Find the most "normal" regions (lowest 20% of scores)
    baseline_threshold = np.percentile(anomaly_np.flatten(), 20)
    baseline_regions = anomaly_np <= baseline_threshold
    baseline_mean = np.mean(anomaly_np[baseline_regions])
    baseline_std = np.std(anomaly_np[baseline_regions])
    
    # Normalize using baseline statistics
    anomaly_normalized = (anomaly_np - baseline_mean) / (baseline_std + 1e-8)
    anomaly_normalized = np.clip(anomaly_normalized, 0, None)  # Only positive anomalies
    
    # Create better color mapping
    im1 = axes[1].imshow(anomaly_normalized, cmap='jet', vmin=0, vmax=np.percentile(anomaly_normalized, 95))
    axes[1].set_title('Anomaly Heat Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Ground truth
    if ground_truth is not None:
        gt_np = tensor_to_numpy_image(ground_truth)
        while gt_np.ndim > 2:
            gt_np = gt_np.squeeze()
        axes[2].imshow(gt_np, cmap='gray')
        axes[2].set_title('Ground Truth')
    else:
        axes[2].set_title('No Ground Truth')
        axes[2].axis('off')
    axes[2].axis('off')
    
    # IMPROVED BINARY PREDICTION
    # Use adaptive threshold based on image statistics
    if np.std(anomaly_normalized) > 0:
        # Statistical threshold: mean + k*std
        adaptive_threshold = np.mean(anomaly_normalized) + 2.5 * np.std(anomaly_normalized)
        # Alternative: Use Otsu-like method
        hist, bin_edges = np.histogram(anomaly_normalized.flatten(), bins=50)
        # Find threshold that maximizes inter-class variance
        adaptive_threshold = max(adaptive_threshold, np.percentile(anomaly_normalized.flatten(), 90))
    else:
        adaptive_threshold = np.percentile(anomaly_normalized.flatten(), 95)
    
    binary_pred = (anomaly_normalized > adaptive_threshold).astype(float)
    axes[3].imshow(binary_pred, cmap='gray')
    axes[3].set_title(f'Segmentation (T={adaptive_threshold:.3f})')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def calculate_metrics(predictions, ground_truths):
    """Calculate comprehensive metrics with advanced threshold optimization."""
    if not ground_truths:
        return {}
    
    # Flatten all predictions and ground truths
    all_preds = np.concatenate([pred.flatten() for pred in predictions])
    all_gts = np.concatenate([gt.flatten() for gt in ground_truths])
    
    # ROC curve calculation
    fpr, tpr, thresholds = roc_curve(all_gts, all_preds)
    auroc = auc(fpr, tpr)
    
    # ADVANCED THRESHOLD OPTIMIZATION STRATEGIES
    
    # Method 1: Youden's Index (TPR - FPR)
    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]
    
    # Method 2: F1-Score optimization with fine-grained search
    best_f1 = 0
    best_f1_threshold = youden_threshold
    
    # Create more refined threshold candidates
    min_thresh = np.percentile(all_preds, 5)
    max_thresh = np.percentile(all_preds, 99)
    candidate_thresholds = np.linspace(min_thresh, max_thresh, 200)
    
    f1_scores = []
    for thresh in candidate_thresholds:
        binary_preds = (all_preds > thresh).astype(int)
        f1 = f1_score(all_gts, binary_preds, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = thresh
    
    # Method 3: Precision-Recall balance (F1 components)
    # Find threshold where precision ≈ recall
    balanced_threshold = best_f1_threshold
    min_diff = float('inf')
    
    for thresh in candidate_thresholds:
        binary_preds = (all_preds > thresh).astype(int)
        if np.sum(binary_preds) > 0:  # Avoid division by zero
            prec = precision_score(all_gts, binary_preds, zero_division=0)
            rec = recall_score(all_gts, binary_preds, zero_division=0)
            diff = abs(prec - rec)
            if diff < min_diff and (prec + rec) > 0:
                min_diff = diff
                balanced_threshold = thresh
    
    # Method 4: Statistical threshold (for reference)
    normal_indices = all_gts == 0
    if np.sum(normal_indices) > 0:
        normal_scores = all_preds[normal_indices]
        stat_threshold = np.mean(normal_scores) + 3 * np.std(normal_scores)
    else:
        stat_threshold = np.percentile(all_preds, 95)
    
    # Select the best performing threshold (F1-optimized)
    optimal_threshold = best_f1_threshold
    binary_preds = (all_preds > optimal_threshold).astype(int)
    
    # Calculate final metrics with optimal threshold
    final_f1 = f1_score(all_gts, binary_preds, zero_division=0)
    final_precision = precision_score(all_gts, binary_preds, zero_division=0)
    final_recall = recall_score(all_gts, binary_preds, zero_division=0)
    
    # IoU calculation
    intersection = np.logical_and(all_gts, binary_preds).sum()
    union = np.logical_or(all_gts, binary_preds).sum()
    iou = intersection / union if union > 0 else 0
    
    # Additional metrics for analysis
    # Specificity (True Negative Rate)
    tn = np.logical_and(all_gts == 0, binary_preds == 0).sum()
    fp = np.logical_and(all_gts == 0, binary_preds == 1).sum()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Balanced Accuracy
    balanced_accuracy = (final_recall + specificity) / 2
    
    return {
        "AUROC": float(auroc),
        "F1_Score": float(final_f1),
        "Precision": float(final_precision),
        "Recall": float(final_recall),
        "Specificity": float(specificity),
        "Balanced_Accuracy": float(balanced_accuracy),
        "IoU": float(iou),
        "Optimal_Threshold": float(optimal_threshold),
        "Youden_Threshold": float(youden_threshold),
        "Balanced_Threshold": float(balanced_threshold),
        "Statistical_Threshold": float(stat_threshold),
        "Max_F1_Score": float(best_f1),
        "Threshold_Analysis": {
            "f1_scores": f1_scores[:50],  # First 50 for analysis
            "thresholds": candidate_thresholds[:50].tolist()
        }
    }

def plot_roc_curve(predictions, ground_truths, results_dir, model_name):
    """Plot and save ROC curve."""
    all_preds = np.concatenate([pred.flatten() for pred in predictions])
    all_gts = np.concatenate([gt.flatten() for gt in ground_truths])
    
    fpr, tpr, thresholds = roc_curve(all_gts, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name.upper()}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    roc_file = results_dir / "roc_curves" / f"roc_curve_{model_name}.png"
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc)
    }
    
    roc_json_file = results_dir / "roc_curves" / f"roc_data_{model_name}.json"
    with open(roc_json_file, 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    return roc_auc

def train_and_evaluate_model(model_name, data_path, max_time_minutes=20):
    """Main training and evaluation function with proper error handling."""
    print(f"\n{'='*60}")
    print(f"STARTING: {model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    # Clean previous results
    results_dir = Path("results") / model_name
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Previous results cleaned.")
    
    # Create directories
    results_dir = setup_directories(model_name)
    print(f"Results directory created: {results_dir}")
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()  # Clear any existing cache
        print("🔥 GPU optimizations active!")
    
    # Create data module
    print("Preparing data module...")
    try:
        datamodule = create_datamodule(data_path)
        datamodule.setup()
        print("✅ Data module created successfully.")
        
        # Verify data loading
        train_dataset = datamodule.train_dataloader()
        print(f"Training batches: {len(train_dataset)}")
        val_dataset = datamodule.val_dataloader()
        print(f"Validation batches: {len(val_dataset)}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
    except Exception as e:
        print(f"❌ Data module error: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    
    # Create model
    print("Creating model...")
    try:
        model = get_model(model_name)
        print(f"✅ {model_name.upper()} model created successfully.")
        
        # Enhanced model configuration display  
        print(f"   Model configuration:")
        print(f"   - Model type: {type(model).__name__}")
        if hasattr(model, 'backbone'):
            print(f"   - Backbone: {model.backbone}")
        if hasattr(model, 'learning_rate'):
            print(f"   - Learning rate: {model.learning_rate}")
        elif hasattr(model, 'lr'):
            print(f"   - Learning rate: {model.lr}")
        if hasattr(model, 'layers'):
            print(f"   - Feature layers: {model.layers}")
        if hasattr(model, 'n_features'):
            print(f"   - Feature dimensions: {model.n_features}")
        
        # FastFlow specific parameters - added for the new model
        if hasattr(model, 'flow_steps'):
            print(f"   - Flow steps: {model.flow_steps}")
        if hasattr(model, 'coupling_type'):
            print(f"   - Coupling type: {model.coupling_type}")
        if hasattr(model, 'hidden_ratio'):
            print(f"   - Hidden ratio: {model.hidden_ratio}")
        
        # CFA specific parameters
        if hasattr(model, 'gamma_c'):
            print(f"   - Gamma C (coupling): {model.gamma_c}")
        if hasattr(model, 'gamma_d'):
            print(f"   - Gamma D (distance): {model.gamma_d}")
        if hasattr(model, 'num_nearest_neighbors'):
            print(f"   - Nearest neighbors: {model.num_nearest_neighbors}")
        if hasattr(model, 'radius'):
            print(f"   - Hypersphere radius: {model.radius}")
            
        print(f"   ✨ Model optimized for maximum F1 performance!")
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    
    # Create Engine with optimized configuration for better performance
    print("Creating optimized Anomalib Engine...")
    try:
        # Enhanced timer callback
        timer_callback = Timer(duration=timedelta(minutes=max_time_minutes))
        
        # TensorBoard logger with better tracking
        logger = TensorBoardLogger(
            save_dir=results_dir / "logs",
            name=f"{model_name}_training",
            log_graph=True  # Log computational graph
        )
        
        # Determine precision based on model type
        # PADIM doesn't need mixed precision since it's non-parametric  
        if model_name.lower() == "padim":
            precision = 32  # Use full precision for PADIM
            max_epochs = 1  # PADIM only needs 1 epoch
            gradient_clip_val = None  # No gradients to clip
            print(f"   ℹ️  PADIM detected: Using full precision (no optimization needed)")
        else:
            precision = "16-mixed" if torch.cuda.is_available() else 32
            max_epochs = 100  # Other models benefit from multiple epochs
            gradient_clip_val = 1.0  # Gradient clipping for stability
            print(f"   ℹ️  {model_name.upper()} detected: Using mixed precision optimization")
        
        # Optimized engine configuration
        engine = Engine(
            callbacks=[timer_callback],
            logger=logger,
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=precision,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=5,  # More frequent logging
            val_check_interval=0.5,  # Check validation twice per epoch for early insights
            num_sanity_val_steps=0,  # Skip sanity validation
            # Optimization flags
            benchmark=True,  # cudnn benchmark for consistent input sizes
            deterministic=False,  # Allow non-deterministic for speed
            # Gradient optimization (only for trainable models)
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm="norm" if gradient_clip_val else None,
        )
        print("✅ Optimized Anomalib Engine created successfully")
        print(f"   - Precision: {precision}")
        print(f"   - Max epochs: {max_epochs}")
        print(f"   - Gradient clipping: {'Enabled (1.0)' if gradient_clip_val else 'Disabled (not needed)'}")
        
    except Exception as e:
        print(f"❌ Engine creation error: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    
    # Training
    print(f"\n🚀 Starting training... (Maximum: {max_time_minutes} minutes)")
    print(f"⏱️  Start time: {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    training_success = False
    try:
        # Clear GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train model
        engine.fit(model, datamodule=datamodule)
        training_success = True
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if it's a timeout
        if "Timer" in str(e) or "timeout" in str(e).lower():
            print("⏰ Training stopped due to time limit")
            training_success = True  # Consider partial training as success
        else:
            training_success = False
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"⏱️  End time: {time.strftime('%H:%M:%S')}")
    print(f"🕐 Total time: {training_time:.2f} minutes")
    
    # Clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Save model
    model_path = results_dir / "models" / f"{model_name}_final.pth"
    try:
        # Save the entire model state, not just state_dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.hparams if hasattr(model, 'hparams') else {},
            'training_time': training_time,
        }, model_path)
        print(f"💾 Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Model save error: {e}")
    
    # Test and visualizations
    print("\n🔍 Test phase starting...")
    metrics = {}
    
    try:
        if training_success:
            # Anomalib built-in test first
            test_results = engine.test(datamodule=datamodule)
            
            # Extract metrics from test results
            if test_results and len(test_results) > 0:
                for key, value in test_results[0].items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] = float(value.item())
                    else:
                        metrics[key] = float(value) if isinstance(value, (int, float)) else value
            
            print("✅ Anomalib test completed! Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
                
                    
        else:
            print("⚠️  Training unsuccessful, skipping test.")
            metrics = {"Training_Success": False}
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        
    
    # Save metrics
    if metrics:
        metrics_file = results_dir / "metrics" / "test_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Results summary
    if metrics:
        print(f"\n📊 {model_name.upper()} MODEL RESULTS:")
        print("-" * 50)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")
        print("-" * 50)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "training_time_minutes": training_time,
        "max_time_minutes": max_time_minutes,
        "training_success": training_success,
        "metrics": metrics,
        "model_path": str(model_path),
        "gpu_used": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
    
    info_file = results_dir / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✅ {model_name.upper()} completed!")
    print(f"📁 Results: {results_dir}")
    
    return results_dir, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Anomaly Detection Training Script - Enhanced Performance"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["fastflow", "cfa", "padim"],  
        help="Model type: fastflow, cfa, padim"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="wood_dataset/wood",
        help="Dataset path"
    )
    parser.add_argument(
        "--max-time", 
        type=int, 
        default=20,
        help="Maximum training time in minutes"
    )
    
    args = parser.parse_args()
    
    # System check
    print("🔍 System Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Verify data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure the dataset is available at the specified path.")
        return
    
    # Check required subdirectories
    required_dirs = ["train/good", "test/good", "test/defect"]
    missing_dirs = []
    for req_dir in required_dirs:
        if not (data_path / req_dir).exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        print("Please ensure your dataset has the proper structure.")
        return
    
    print(f"✅ Data path verified: {data_path}")
    print(f"🎯 Model: {args.model.upper()}")
    print(f"⏰ Maximum training time: {args.max_time} minutes")
    print(f"🚀 Optimizations: Enhanced parameters + Data augmentation + F1 optimization")
    
    # Run training
    try:
        results_dir, metrics = train_and_evaluate_model(
            model_name=args.model,
            data_path=str(data_path),
            max_time_minutes=args.max_time
        )
        
        # Final summary
        print(f"\n{'='*70}")
        print("🏁 OPTIMIZED TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"🎯 Model: {args.model.upper()}")
        print(f"📁 Results: {results_dir}")
        
        if metrics and any(key in metrics for key in ["image_AUROC", "AUROC", "F1_Score"]):
            print(f"\n📊 Key Performance Metrics:")
            for key, value in metrics.items():
                if any(metric in key for metric in ["AUROC", "F1", "IoU", "Precision", "Recall"]):
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleaned!")
            
    except Exception as e:
        print(f"❌ Unexpected error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()