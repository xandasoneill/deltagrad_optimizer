from ultralytics import YOLO
from DeltaGrad import DeltaGrad

# 1. Define a factory function to handle custom hyperparameters
def deltagrad_factory(params):
    return DeltaGrad(
        params, 
        lr=0.01,       # Base learning rate
        K=5,           # History window size
        alpha=0.15,    # Reliability weight
        beta=0.85,     # Inertia weight
        gamma=0.9,     # Scaling factor
        smoothing=0.9  # Gradient EMA factor
    )

# 2. Initialize the YOLO model
model = YOLO('yolov11n.yaml')

# 3. Start training with the custom optimizer
model.train(
    data='coco.yaml', 
    optimizer=deltagrad_factory, 
    epochs=100,
    imgsz=640
)