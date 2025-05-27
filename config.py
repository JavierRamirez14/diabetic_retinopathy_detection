import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T

DATA_DIR = './'
TRAIN_PATH = os.path.join(DATA_DIR, 'dataset/train_320')
VAL_PATH = os.path.join(DATA_DIR, 'dataset/val_320')
TEST_PATH = os.path.join(DATA_DIR, 'dataset/test_320')

TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'dataset/train_labels.csv')
VAL_LABELS_PATH = os.path.join(DATA_DIR, 'dataset/val_labels.csv')

MODEL_DIR = os.path.join(DATA_DIR, 'models')
BEST_EFFICIENTNET_MODEL = os.path.join(MODEL_DIR, 'best_efficientnet_model.pth')
BEST_ENSEMBLE_MODEL = os.path.join(MODEL_DIR, 'best_multi_eye_model.pth')

SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')
FIGURES_PATH = os.path.join(DATA_DIR, 'figures')

LOAD_MODEL = True

# Configuración de entrenamiento
IMG_SIZE = (40, 40)
BATCH_SIZE = 64
NUM_CLASSES = 5
LEARNING_RATE_EF = 1e-4
WEIGHT_DECAY_EF = 5e-4
LEARNING_RATE_TE = 1e-5
WEIGHT_DECAY_TE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.3762, 0.2616, 0.1867]
STD = [0.2518, 0.1773, 0.1287]

EPOCHS_EFFICIENTNET = 0
EPOCHS_TWO_EYES = 0
PATIENCE_EPOCHS = 10

# Transformaciones
train_transform = T.Compose([
    T.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=30, interpolation=InterpolationMode.BILINEAR),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# Val y Test usan la misma transformación
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])