import os
try:
    import torch
except ImportError:
    torch = None

class Config:
    # Data Paths
    DATA_ROOT = r"D:\Downloads\BrainTumorDataSet\preprocessed"
    SPLIT_DIR = r"D:\Downloads\MedDA-Old2Modern-main\MedDA-Old2Modern-main\config" # Path to pickles
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # Model Hyperparameters
    IN_CHANNELS = 4
    OUT_CHANNELS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    
    # Training Environment
    DEVICE = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    # Research Settings
    USE_AUGMENTATION = True # Toggle Hybrid Degradation
    SAVE_MODEL_PATH = "unetplusplus_brain_tumor.pth"
