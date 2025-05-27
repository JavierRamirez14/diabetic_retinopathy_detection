import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score
from scipy.optimize import minimize
import config

def get_preds_and_labels(loader, model, device):
    model.eval()

    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Obteniendo predicciones", leave=False)

    with torch.no_grad():
        for batch_idx, (data, labels, _) in enumerate(loop):
            data = data.to(device)
            labels = labels.float().to(device)

            outputs = model(data)
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    return all_preds, all_labels

def get_metrics(loader, model, loss_fn, device, model_type):
    preds, labels = get_preds_and_labels(loader, model, device)

    if model_type == "EfficientNet":
        labels = labels.view(-1, 1)
    else:
        preds = preds.flatten().view(-1, 1)
        labels = labels.flatten().view(-1, 1)

    loss = loss_fn(preds, labels)

    preds = torch.clamp(preds, -0.5, 4.5)
    preds = torch.round(preds).long().squeeze(1)

    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')

    return loss, f1, qwk

def optimize_prediction_thresholds(raw_preds, true_labels):
    if isinstance(raw_preds, torch.Tensor):
        raw_preds = raw_preds.numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.numpy()

    valid_indices = true_labels != -1
    raw_preds = raw_preds[valid_indices]
    true_labels = true_labels[valid_indices]
    if len(true_labels) == 0:
        print("No hay etiquetas válidas para optimizar umbrales.")
        return np.array([0.5, 1.5, 2.5, 3.5])

    raw_preds = raw_preds.astype(np.float32)
    true_labels = true_labels.astype(np.int32)

    def _qwk_loss(thresholds):
        current_thresholds = np.sort(thresholds)
        cutoffs = np.concatenate(([-np.inf], current_thresholds, [np.inf]))
        pred_labels = np.digitize(raw_preds, cutoffs) - 1
        pred_labels = np.clip(pred_labels, 0, config.NUM_CLASSES - 1)
        
        return -cohen_kappa_score(true_labels, pred_labels, weights='quadratic')

    initial_thresholds = [i + 0.5 for i in range(config.NUM_CLASSES - 1)]
    
    min_val_pred = np.min(raw_preds) if len(raw_preds) > 0 else 0
    max_val_pred = np.max(raw_preds) if len(raw_preds) > 0 else (config.NUM_CLASSES - 1)
    bounds = [(min_val_pred - 1, max_val_pred + 1)] * (config.NUM_CLASSES - 1)

    result = minimize(_qwk_loss, initial_thresholds, method='Powell', bounds=bounds, tol=1e-5)
    
    best_thresholds = np.sort(result.x)
    return best_thresholds

def apply_thresholds_to_predictions(raw_preds, thresholds):
    if isinstance(raw_preds, torch.Tensor):
        raw_preds = raw_preds.numpy()
    
    sorted_thresholds = np.sort(thresholds)
    cutoffs = np.concatenate(([-np.inf], sorted_thresholds, [np.inf]))
    
    pred_classes = np.digitize(raw_preds, cutoffs) - 1
    pred_classes = np.clip(pred_classes, 0, config.NUM_CLASSES - 1)
    return pred_classes

def get_submission(loader, model, thresholds, names, device):
    preds, _ = get_preds_and_labels(loader, model, device)

    preds = preds.flatten().view(-1, 1)
    preds = apply_thresholds_to_predictions(preds, thresholds)
    preds.flatten()

    rows = []

    for idx, name in enumerate(names):
        left_pred = preds[2 * idx].item()
        right_pred = preds[2 * idx + 1].item()

        rows.append({'image': f'{name}_left', 'level': left_pred})
        rows.append({'image': f'{name}_right', 'level': right_pred})

    return pd.DataFrame(rows)
