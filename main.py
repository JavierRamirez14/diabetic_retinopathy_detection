import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

import config
from datasets import DRDataset, EyePairDataset
from training_utils import train, load_checkpoint
from evaluating_utils import get_metrics, optimize_prediction_thresholds, apply_thresholds_to_predictions, get_submission, get_preds_and_labels
from twoeyesmodel_and_utils import extract_features, join_features, TwoEyesModel
from visualization_utils import plot_training_history, plot_validation_results


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_dataset = DRDataset(config.TRAIN_PATH, config.TRAIN_LABELS_PATH, transform=config.train_transform)
    val_dataset = DRDataset(config.VAL_PATH, config.VAL_LABELS_PATH, transform=config.val_transform)
    test_dataset = DRDataset(config.TEST_PATH, train=False, transform=config.val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # EfficientNet Model
    model = EfficientNet.from_pretrained('efficientnet-b3')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_EF, weight_decay=config.WEIGHT_DECAY_EF)

    if config.LOAD_MODEL:
        model, optimizer, best_metric = load_checkpoint(config.BEST_EFFICIENTNET_MODEL, model, optimizer, config.DEVICE)
        print(f"Modelo cargado desde {config.BEST_EFFICIENTNET_MODEL} con QWK: {best_metric:.4f}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.MSELoss()

    history_ef, model = train(train_loader, val_loader, model, config.EPOCHS_EFFICIENTNET, optimizer, scheduler, loss_fn, scaler, config.PATIENCE_EPOCHS, config.BEST_EFFICIENTNET_MODEL, config.DEVICE, "EfficientNet")

    # Extraer features
    model_feature = copy.deepcopy(model)
    model_feature._fc = nn.Identity()
    
    train_preds, train_labels, train_image_names = extract_features(train_loader, model_feature, config.DEVICE)
    val_preds, val_labels, val_image_names = extract_features(val_loader, model_feature, config.DEVICE)
    test_preds, test_labels, test_image_names = extract_features(test_loader, model_feature, config.DEVICE)

    train_preds, train_labels, _ = join_features(train_image_names, train_preds, train_labels)
    val_preds, val_labels, _ = join_features(val_image_names, val_preds, val_labels)
    test_preds, test_labels, test_id_names = join_features(test_image_names, test_preds, test_labels)

    # Datasets dos ojos
    train_dataset_te = EyePairDataset(train_preds, train_labels)
    val_dataset_te = EyePairDataset(val_preds, val_labels)
    test_dataset_te = EyePairDataset(test_preds, test_labels)

    train_loader_te = DataLoader(train_dataset_te, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader_te = DataLoader(val_dataset_te, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader_te = DataLoader(test_dataset_te, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # Entrenamiento modelo dos ojos
    model_te = TwoEyesModel(input_dim=in_features*2)
    model_te = model_te.to(config.DEVICE)
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=config.LEARNING_RATE_TE, weight_decay=config.WEIGHT_DECAY_TE)
    scheduler_te = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_te, mode='max', factor=0.1, patience=5)
    scaler_te = torch.amp.GradScaler('cuda')
    loss_fn_te = nn.MSELoss()

    history_te, model_te = train(train_loader_te, val_loader_te, model_te, config.EPOCHS_TWO_EYES, optimizer_te, scheduler_te, loss_fn_te, scaler_te, config.PATIENCE_EPOCHS, config.BEST_TWO_EYES_MODEL, config.DEVICE, "TwoEyes")

    # Ajustar thresholds
    preds, labels = get_preds_and_labels(val_loader_te, model_te, config.DEVICE)
    preds = preds.flatten().view(-1,1)
    labels = labels.flatten().view(-1,1)

    thresholds = optimize_prediction_thresholds(preds.cpu().numpy(), labels.cpu().numpy())
    preds_th = apply_thresholds_to_predictions(preds, thresholds)
    acc = (preds_th == labels).float().mean()
    qwk = cohen_kappa_score(labels, preds_th, weights='quadratic')
    print(f"QWK con thresholds optimizados: {qwk}")

    os.makedirs(config.FIGURES_PATH, exist_ok=True)
    plot_training_history(history_ef, "EfficientNet")
    plot_training_history(history_te, "Two Eyes")
    plot_validation_results(preds, preds_th, labels)

    submission = get_submission(test_loader_te, model_te, thresholds, test_id_names, config.DEVICE)
    submission.to_csv(config.SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    main()
