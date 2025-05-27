import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Dataset para las imágenes iniciales
class DRDataset(Dataset):
    def __init__(self, path_images, path_labels=None, train=True, transform=None):
        super().__init__()
        self.path_images = path_images
        self.train = train
        self.transform = transform

        if self.train:
            self.data = pd.read_csv(path_labels)
            self.image_files = self.data['image'].tolist()
            self.labels = self.data['level'].tolist()
        else:
            self.image_files = sorted([
                f for f in os.listdir(self.path_images)
                if f.lower().endswith(('.jpeg', '.jpg', '.png'))
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.train:
            image_name = self.image_files[idx]
            label = self.labels[idx]
        else:
            image_name = os.path.splitext(self.image_files[idx])[0]
            label = -1

        image_path = os.path.join(self.path_images, image_name + '.jpeg')
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = np.array(image)

        return image, label, image_name


# Dataset para las características de los dos ojos
class EyePairDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], ''