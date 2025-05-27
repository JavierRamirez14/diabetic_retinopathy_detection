import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm

def extract_features(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_image_names = []

    with torch.no_grad():
        for data, label, image_names in tqdm(loader):
            data = data.to(device)
            outputs = model(data)
            all_preds.append(outputs.cpu())
            all_labels.append(label.cpu())

            all_image_names.extend(image_names)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return all_preds, all_labels, all_image_names

def join_features(image_names, preds, labels):
    features_dict = defaultdict(dict)

    for name, feature, label in zip(image_names, preds, labels):
        patient_id, eye = name.split('_')
        features_dict[patient_id][eye] = feature
        features_dict[patient_id]['label_' + eye] = label

    X = []
    y = []

    for patient_id, data in features_dict.items():
        if 'left' in data and 'right' in data:
            combined = torch.cat([data['left'], data['right']])
            X.append(combined)

            labels_combined = torch.stack([data['label_left'], data['label_right']])
            y.append(labels_combined)

    X = torch.stack(X)
    y = torch.stack(y)
    image_names = list(features_dict.keys())

    return X, y, image_names

class TwoEyesModel(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256, dropout_val=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout_val)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.output(x)
