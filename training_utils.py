import os
import torch
import numpy as np
from tqdm import tqdm
import config
import evaluating_utils

# Función para guardar el checkpoint del modelo
def save_checkpoint(model, optimizer, best_qwk, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_qwk': best_qwk
    }, filepath)

# Función para cargar el checkpoint del modelo
def load_checkpoint(filepath, model, optimizer=None, device=config.DEVICE):
    if not os.path.exists(filepath):
        print(f"Checkpoint no encontrado en {filepath}, no se carga nada.")
        return model, optimizer, -float('inf')

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_metric = checkpoint.get('best_qwk', -float('inf'))

    return model, optimizer, best_metric

# Función de entrenamiento de una época
def train_epoch(loader, model, optimizer, loss_fn, scaler, device, model_type="EfficientNet"):
    model.train()
    epoch_losses = []

    loop_desc = f"Entrenando Epoch ({model_type})"
    loop = tqdm(loader, desc=loop_desc)

    for batch_idx, (data, labels, _) in enumerate(loop):
        data = data.to(device)
        # Para EfficientNet (single output): labels a [B, 1] y float
        # Para Ensemble (dos outputs): labels a [B, 2] y float
        if model_type == "EfficientNet":
            labels = labels.float().unsqueeze(1).to(device)
        else:  # Ensemble
            labels = labels.float().to(device)

        # Forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(data)
            loss = loss_fn(outputs, labels)

        epoch_losses.append(loss.item())

        # Backward pass y optimización
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    return np.mean(epoch_losses)

# Función de entrenamiento completa
def train(train_loader, val_loader, model, epochs, optimizer, scheduler, loss_fn, scaler, patience_epochs, ruta, device, model_type="Efficientnet"):
    print(f'Iniciando entrenamiento {model_type}')

    history = {'train_loss': [], 'val_loss': [], 'train_qwk': [], 'val_qwk': [], 'train_f1': [], 'val_f1': []}

    best_qwk = -float('inf')
    patience = 0

    for epoch in range(epochs):
        loss = train_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, model_type)

        train_loss, train_f1, train_qwk = evaluating_utils.get_metrics(train_loader, model, loss_fn, config.DEVICE, model_type)
        val_loss, val_f1, val_qwk = evaluating_utils.get_metrics(val_loader, model, loss_fn, config.DEVICE, model_type)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_qwk'].append(train_qwk)
        history['val_qwk'].append(val_qwk)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        scheduler.step(val_qwk)

        print(f'Epoch: {epoch+1}/{epochs} --> Train Loss: {train_loss} | Val Loss: {val_loss} | Train QWK: {train_qwk} | Val QWK: {val_qwk}')

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience = 0
            save_checkpoint(model, optimizer, best_qwk, ruta)
        else:
            patience += 1
            if patience >= patience_epochs:
                print(f'Early stopping en epoch {epoch+1}')
                break

    model, _, loaded_qwk = load_checkpoint(ruta, model, optimizer, config.DEVICE)

    print(f'--- Entrenamiento {model_type} terminado con qwk: {loaded_qwk} ---')

    return history, model
