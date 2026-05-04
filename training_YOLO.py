from ultralytics import YOLO
from DeltaGrad import DeltaGrad
import os
import torch
import threading

def async_save(data, path):
    """Saves the metric dictionary to disk in a separate thread to avoid blocking."""
    torch.save(data, path)

def deltagrad_factory(params):
    """Factory function to instantiate DeltaGrad with custom hyperparameters."""
    return DeltaGrad(
        params, 
        lr=0.01,       # Base learning rate
        K=4,           # History window size
        alpha=0.1,     # Reliability weight
        beta=0.9,      # Inertia weight
        gamma=0.9,     # Global scaling factor
        smoothing=0.9  # Gradient EMA factor
    )

def callback_save_deltagrad_state(trainer):
    """
    Extracts weights, gradients, and R metrics at the end of each epoch 
    and saves them asynchronously to ensure accurate training time reporting.
    """
    epoch = trainer.epoch
    optimizer = trainer.optimizer
    save_dir = "deltagrad_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    stats = {}
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            # Move data to CPU and convert to FP16 to minimize RAM usage and save time
            data = {
                'w': param.data.cpu().half(), 
                'g': param.grad.data.cpu().half() if param.grad is not None else None
            }
            
            # Capture the internal DeltaGrad reliability metric (R)
            if param in optimizer.state and 'R' in optimizer.state[param]:
                data['R'] = optimizer.state[param]['R'].cpu().half()
            
            stats[name] = data
    
    # Trigger the background thread for saving to disk
    save_path = f"{save_dir}/epoch_{epoch}.pt"
    thread = threading.Thread(target=async_save, args=(stats, save_path))
    thread.start()

# 2. Initialize the YOLO model
model = YOLO('yolov8n.yaml')

model.add_callback("on_train_epoch_end", callback_save_deltagrad_state)

# 3. Start training with the custom optimizer
model.train(
    data='coco.yaml', 
    optimizer=deltagrad_factory, 
    epochs=100,
    imgsz=640
)