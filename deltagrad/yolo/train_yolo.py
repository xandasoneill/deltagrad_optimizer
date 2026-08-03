from ultralytics import YOLO
from deltagrad.optimizers import DeltaGradWindowedLegacy as DeltaGrad
import os
import torch
import threading
from ultralytics.engine.trainer import BaseTrainer


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
            data = {
                'w': param.data.cpu().half(),
                'g': param.grad.data.cpu().half() if param.grad is not None else None
            }

            if param in optimizer.state and 'R' in optimizer.state[param]:
                data['R'] = optimizer.state[param]['R'].cpu().half()

            stats[name] = data

    save_path = f"{save_dir}/epoch_{epoch}.pt"
    thread = threading.Thread(target=async_save, args=(stats, save_path))
    thread.start()


original_build_optimizer = BaseTrainer.build_optimizer


def custom_build_optimizer(self, *args, **kwargs):

    if self.args.optimizer == 'deltagrad':

        return deltagrad_factory(self.model.parameters())
    else:

        return original_build_optimizer(self, *args, **kwargs)


BaseTrainer.build_optimizer = custom_build_optimizer

if __name__ == "__main__":
    model = YOLO('yolov8n.yaml')

    model.add_callback("on_train_epoch_end", callback_save_deltagrad_state)

    model.train(
        data='coco.yaml',
        optimizer='deltagrad',
        epochs=100,
        imgsz=640,
        batch=32,
        project='/content/runs',
        workers=8
    )
