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
from anomalib.models import Cfa
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
    """Create data module with proper configuration."""
    # More conservative batch sizes to avoid memory issues
    batch_size = 8 if torch.cuda.is_available() else 4
    num_workers = 2 if torch.cuda.is_available() else 1
    
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
        val_split_ratio=0.2,
        val_split_mode="from_train",  
        seed=42
    )
    return datamodule

def get_model(model_name):
    """Get model based on model name with correct parameters."""
    if model_name.lower() == "cfa":
        model = Cfa(
            backbone="wide_resnet50_2",
            gamma_c=1,
            gamma_d=1,
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=1e-05
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only 'cfa' is supported.")
    
    # Set learning rate manually if the model supports it
    if hasattr(model, 'learning_rate'):
        model.learning_rate = 0.0001
    elif hasattr(model, 'lr'):
        model.lr = 0.0001
    
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
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    img_np = tensor_to_numpy_image(image)
    if img_np.ndim == 2:
        axes[0].imshow(img_np, cmap='gray')
    else:
        axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Anomaly map
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_np = anomaly_map.cpu().numpy()
    else:
        anomaly_np = anomaly_map
    
    # Remove extra dimensions
    while anomaly_np.ndim > 2:
        anomaly_np = anomaly_np.squeeze()
    
    im1 = axes[1].imshow(anomaly_np, cmap='jet')
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
    
    # Binary prediction
    threshold = np.percentile(anomaly_np.flatten(), 95)
    binary_pred = (anomaly_np > threshold).astype(float)
    axes[3].imshow(binary_pred, cmap='gray')
    axes[3].set_title('Segmentation Result')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def calculate_metrics(predictions, ground_truths):
    """Calculate comprehensive metrics."""
    if not ground_truths:
        return {}
    
    # Flatten all predictions and ground truths
    all_preds = np.concatenate([pred.flatten() for pred in predictions])
    all_gts = np.concatenate([gt.flatten() for gt in ground_truths])
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(all_gts, all_preds)
    auroc = auc(fpr, tpr)
    
    # Determine threshold
    threshold = np.percentile(all_preds, 95)
    binary_preds = (all_preds > threshold).astype(int)
    
    # Classification metrics
    f1 = f1_score(all_gts, binary_preds, zero_division=0)
    precision = precision_score(all_gts, binary_preds, zero_division=0)
    recall = recall_score(all_gts, binary_preds, zero_division=0)
    
    # IoU calculation
    intersection = np.logical_and(all_gts, binary_preds).sum()
    union = np.logical_or(all_gts, binary_preds).sum()
    iou = intersection / union if union > 0 else 0
    
    return {
        "AUROC": float(auroc),
        "F1_Score": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "IoU": float(iou),
        "Threshold": float(threshold)
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
        
        # Print model configuration
        print(f"   Model configuration:")
        if hasattr(model, 'backbone'):
            print(f"   - Backbone: {model.backbone}")
        if hasattr(model, 'learning_rate'):
            print(f"   - Learning rate: {model.learning_rate}")
        elif hasattr(model, 'lr'):
            print(f"   - Learning rate: {model.lr}")
        if hasattr(model, 'layers'):
            print(f"   - Feature layers: {model.layers}")
        
        # CFA specific parameters
        if hasattr(model, 'gamma_c'):
            print(f"   - Gamma C: {model.gamma_c}")
        if hasattr(model, 'gamma_d'):
            print(f"   - Gamma D: {model.gamma_d}")
        if hasattr(model, 'num_nearest_neighbors'):
            print(f"   - Nearest neighbors: {model.num_nearest_neighbors}")
        if hasattr(model, 'radius'):
            print(f"   - Radius: {model.radius}")
            
        print(f"   - Model type: {type(model).__name__}")
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    
    # Create Engine with proper configuration
    print("Creating Anomalib Engine...")
    try:
        # Simple timer callback only
        timer_callback = Timer(duration=timedelta(minutes=max_time_minutes))
        
        # TensorBoard logger
        logger = TensorBoardLogger(
            save_dir=results_dir / "logs",
            name=f"{model_name}_training"
        )
        
        # Create engine with minimal callbacks to avoid conflicts
        engine = Engine(
            callbacks=[timer_callback],  # Only timer callback
            logger=logger,
            max_epochs=50,  # Reasonable number of epochs
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=32,
            enable_checkpointing=True,  # Enable for proper functionality
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            val_check_interval=1.0,  # Check validation every epoch
            num_sanity_val_steps=0,  # Skip sanity validation to avoid empty embeddings error
        )
        print("✅ Anomalib Engine created successfully")
        
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
    
    # Testing and evaluation
    print("\n🔍 Starting test phase...")
    metrics = {}
    
    try:
        if training_success:
            # Test with Anomalib's built-in test method
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
        
        # Try manual evaluation as fallback
        print("📊 Attempting manual evaluation...")
        try:
            metrics = perform_manual_evaluation(model, datamodule, results_dir, model_name)
        except Exception as manual_e:
            print(f"❌ Manual evaluation also failed: {manual_e}")
            metrics = {"Training_Success": training_success}
    
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

def perform_manual_evaluation(model, datamodule, results_dir, model_name):
    """Perform manual evaluation with visualizations."""
    print("Starting manual evaluation and visualization creation...")
    
    model.eval()
    test_dataloader = datamodule.test_dataloader()
    
    all_predictions = []
    all_ground_truths = []
    image_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            try:
                print(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}...")
                
                # Extract images and masks from batch
                images, masks = extract_image_and_mask_from_batch(batch)
                if images is None:
                    print(f"  Skipping batch {batch_idx} - no images found")
                    continue
                
                # Move to device
                if torch.cuda.is_available():
                    batch = {key: value.cuda() if torch.is_tensor(value) else value 
                            for key, value in batch.items()} if isinstance(batch, dict) else batch
                
                # Forward pass
                try:
                    outputs = model(batch)
                    if "anomaly_map" not in outputs:
                        print(f"  Warning: No anomaly_map in outputs for batch {batch_idx}")
                        continue
                except Exception as forward_error:
                    print(f"  Forward pass error in batch {batch_idx}: {forward_error}")
                    continue
                
                # Process each image in the batch
                batch_size = images.shape[0]
                for i in range(min(batch_size, 5)):  # Limit to 5 images per batch for speed
                    image_count += 1
                    
                    try:
                        # Extract single image data
                        single_image = images[i]
                        single_anomaly = outputs["anomaly_map"][i]
                        single_mask = masks[i] if masks is not None else None
                        
                        # Convert to numpy for processing
                        anomaly_np = single_anomaly.cpu().numpy()
                        while anomaly_np.ndim > 2:
                            anomaly_np = anomaly_np.squeeze()
                        
                        all_predictions.append(anomaly_np)
                        
                        if single_mask is not None:
                            mask_np = single_mask.cpu().numpy()
                            while mask_np.ndim > 2:
                                mask_np = mask_np.squeeze()
                            all_ground_truths.append(mask_np)
                        
                        # Create visualizations
                        try:
                            # Full visualization
                            fig = create_visualization(
                                single_image,
                                anomaly_np,
                                single_mask
                            )
                            vis_file = results_dir / "visualizations" / f"visualization_{image_count:03d}.png"
                            fig.savefig(vis_file, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            
                            # Heat map
                            if anomaly_np.max() > anomaly_np.min():
                                heat_map_norm = ((anomaly_np - anomaly_np.min()) / 
                                               (anomaly_np.max() - anomaly_np.min()) * 255).astype(np.uint8)
                            else:
                                heat_map_norm = np.zeros_like(anomaly_np, dtype=np.uint8)
                            
                            heat_map_colored = cv2.applyColorMap(heat_map_norm, cv2.COLORMAP_JET)
                            heatmap_file = results_dir / "heatmaps" / f"heatmap_{image_count:03d}.png"
                            cv2.imwrite(str(heatmap_file), heat_map_colored)
                            
                            # Binary segmentation
                            threshold = np.percentile(anomaly_np, 95) if anomaly_np.max() > anomaly_np.min() else 0.5
                            seg_mask = (anomaly_np > threshold).astype(np.uint8) * 255
                            seg_file = results_dir / "segmentations" / f"segmentation_{image_count:03d}.png"
                            cv2.imwrite(str(seg_file), seg_mask)
                            
                        except Exception as viz_error:
                            print(f"    Visualization error for image {image_count}: {viz_error}")
                            
                    except Exception as image_error:
                        print(f"    Image processing error {image_count}: {image_error}")
                        continue
                        
            except Exception as batch_error:
                print(f"  Batch processing error {batch_idx}: {batch_error}")
                continue
    
    print(f"✅ Processed {image_count} images.")
    
    # Calculate metrics
    metrics = {"Total_Processed_Images": image_count}
    
    if all_predictions and all_ground_truths:
        manual_metrics = calculate_metrics(all_predictions, all_ground_truths)
        roc_auc = plot_roc_curve(all_predictions, all_ground_truths, results_dir, model_name)
        manual_metrics["Manual_ROC_AUC"] = roc_auc
        metrics.update(manual_metrics)
        print("✅ Manual metrics calculated successfully.")
    else:
        print("⚠️  No ground truth available for metric calculation.")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Fixed Anomalib Training Script - CFA Model Only"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="cfa",
        choices=["cfa"],
        help="Model type: only 'cfa' is supported"
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
    print(f"🎯 Model: CFA")
    print(f"⏰ Maximum training time: {args.max_time} minutes")
    
    # Run training
    try:
        results_dir, metrics = train_and_evaluate_model(
            model_name="cfa",
            data_path=str(data_path),
            max_time_minutes=args.max_time
        )
        
        # Final summary
        print(f"\n{'='*70}")
        print("🏁 TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"🎯 Model: CFA")
        print(f"📁 Results: {results_dir}")
        
        if metrics and any(key in metrics for key in ["image_AUROC", "AUROC", "F1_Score"]):
            print(f"\n📊 Key Performance Metrics:")
            for key, value in metrics.items():
                if any(metric in key for metric in ["AUROC", "F1", "IoU", "Precision", "Recall"]):
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        
        print(f"\n📋 Generated Files:")
        print(f"   ├── 🤖 models/           (Trained model weights)")
        print(f"   ├── 📈 metrics/          (Evaluation metrics)")  
        print(f"   ├── 🖼️  visualizations/   (Full image comparisons)")
        print(f"   ├── 📊 roc_curves/       (ROC analysis)")
        print(f"   ├── 🔥 heatmaps/         (Anomaly heat maps)")
        print(f"   └── 🎭 segmentations/    (Binary masks)")
        print(f"{'='*70}")
        
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