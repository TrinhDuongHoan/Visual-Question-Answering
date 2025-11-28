import torch
import os

class Config:

    TRAIN_IMAGES_PATH = '/kaggle/input/openvivqa/data/train-images/training-images'
    TEST_IMAGES_PATH  = '/kaggle/input/openvivqa/data/test-images/test-images'
    DEV_IMAGES_PATH   = '/kaggle/input/openvivqa/data/dev-images/dev-images'
    
    TRAIN_JSON_PATH   = '/kaggle/input/openvivqa/data/vlsp2023_train_data.json'
    TEST_JSON_PATH    = '/kaggle/input/openvivqa/data/vlsp2023_test_data.json'
    DEV_JSON_PATH     = '/kaggle/input/openvivqa/data/vlsp2023_dev_data.json'
    
    # Output paths
    WORKING_DIR     = "/kaggle/working/data"
    TRAIN_JSON_FLAT = os.path.join(WORKING_DIR, "train_flat.json")
    DEV_JSON_FLAT   = os.path.join(WORKING_DIR, "val_flat.json")
    TEST_JSON_FLAT  = os.path.join(WORKING_DIR, "test_flat.json")
    
    CHECKPOINT_DIR = "checkpoints"
    
    # ==== TRAINING CONFIG ====
    SEED = 42
    IMAGE_SIZE = 224
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    NUM_EPOCHS = 10
    EARLY_STOP_PATIENCE = 3
    
    LR_VIT_PHOBERT = 2e-5      
    LR_DECODER = 5e-4          
    WEIGHT_DECAY = 1e-5
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== MODEL CONFIG ====
    TEXT_ENCODER_NAME = "vinai/phobert-base-v2"
    VISION_NAME = "vit_base_patch16_224"
    MAX_QUESTION_LEN = 64
    MAX_ANSWER_LEN = 15
    DEC_HIDDEN_SIZE = 256
    MIN_ANSWER_FREQ = 3
    NUM_BEAMS = 3