import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    f1_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import config

# Global style config
plt.rcParams.update({
    'font.family': 'Century Gothic',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})

accessible_palette = sns.color_palette("colorblind")
training_color = accessible_palette[0]
validation_color = accessible_palette[1]
blue_palette = sns.color_palette("Blues", 5)


def plot_training_history(history, model):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    title = f'{model} Model Training Progress Over Epochs'
    fig.suptitle(title, fontweight='bold', fontsize=20, y=1.03)

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], color=training_color, linewidth=3, marker='o', markersize=6, label='Training')
    axes[0].plot(epochs, history['val_loss'], color=validation_color, linewidth=3, marker='s', markersize=6, label='Validation')
    axes[0].set_title('Loss', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_axisbelow(True)

    axes[1].plot(epochs, history['train_qwk'], color=training_color, linewidth=3, marker='o', markersize=6, label='Training')
    axes[1].plot(epochs, history['val_qwk'], color=validation_color, linewidth=3, marker='s', markersize=6, label='Validation')
    axes[1].set_title('Quadratic Weighted Kappa (QWK)', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('QWK', fontsize=14)
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_axisbelow(True)

    axes[2].plot(epochs, history['train_f1'], color=training_color, linewidth=3, marker='o', markersize=6, label='Training')
    axes[2].plot(epochs, history['val_f1'], color=validation_color, linewidth=3, marker='s', markersize=6, label='Validation')
    axes[2].set_title('F1 Score', fontsize=18, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=14)
    axes[2].set_ylabel('F1 Score', fontsize=14)
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(config.FIGURES_PATH, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_validation_results(preds, preds_th, labels):
    preds = np.array(preds).flatten()
    preds_th = np.array(preds_th).flatten().astype(int)
    labels = np.array(labels).flatten().astype(int)

    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']
    num_classes = len(class_names)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, preds_th)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16, "weight": "bold"}, cbar=False, linewidths=0.8, linecolor='gray')
    plt.title('Confusion Matrix of Predicted vs Actual Classes', fontweight='bold', fontsize=18)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('Actual Class', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Class Distribution Bar Chart
    pred_counts = np.bincount(preds_th, minlength=num_classes)
    true_counts = np.bincount(labels, minlength=num_classes)
    x = np.arange(num_classes)
    width = 0.35

    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, true_counts, width, label='Actual', color=accessible_palette[0], edgecolor='black')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color=accessible_palette[1], edgecolor='black')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Actual vs Predicted Classes', fontsize=18, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ROC Curves
    y_bin = label_binarize(labels, classes=range(num_classes))
    if preds.ndim == 1:
        y_score = np.zeros((len(preds), num_classes))
        for i, val in enumerate(preds):
            for j in range(num_classes):
                y_score[i, j] = np.exp(-0.5 * ((val - j) ** 2))
        y_score /= y_score.sum(axis=1, keepdims=True)
    else:
        y_score = preds

    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if y_bin.shape[1] > i and y_score.shape[1] > i:
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=3, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})',
                     color=accessible_palette[i], marker='o', markevery=0.1, markersize=6)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for Each Class (One-vs-Rest)', fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Precision, Recall and F1 Score per Class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_th, average=None, zero_division=0)
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(x - width, precision, width, label='Precision', color=accessible_palette[0], edgecolor='black')
    plt.bar(x, recall, width, label='Recall', color=accessible_palette[1], edgecolor='black')
    plt.bar(x + width, f1, width, label='F1 Score', color=accessible_palette[2], edgecolor='black')
    plt.xticks(x, class_names, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.05)
    plt.title('Precision, Recall, and F1 Score per Class', fontsize=18, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_PATH, 'precision_recall_f1_per_class.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Global Metrics Table
    metrics_data = {
        'Metric': ['Accuracy', 'Macro F1', 'Weighted F1', 'QWK'],
        'Value': [
            f"{accuracy_score(labels, preds_th):.4f}",
            f"{np.mean(f1):.4f}",
            f"{f1_score(labels, preds_th, average='weighted'):.4f}",
            f"{cohen_kappa_score(labels, preds_th, weights='quadratic'):.4f}"
        ]
    }
    df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.5, 2.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_fontsize(16)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#0072B2')
        else:
            cell.set_facecolor('#D0E1F9' if key[0] % 2 == 1 else 'white')
            if key[1] == 0:
                cell.set_text_props(weight='bold')
    plt.title('Summary of Global Metrics', fontweight='bold', fontsize=18, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_PATH, 'global_metrics_table.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Detailed Classification Report Table
    report = classification_report(labels, preds_th, target_names=class_names, output_dict=True)
    rows = []
    for name in class_names:
        row = report.get(name, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        rows.append([name, f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1-score']:.3f}", int(row['support'])])

    if 'macro avg' in report:
        rows.append(['Macro Avg', f"{report['macro avg']['precision']:.3f}", f"{report['macro avg']['recall']:.3f}", f"{report['macro avg']['f1-score']:.3f}", int(report['macro avg']['support'])])
    if 'weighted avg' in report:
        rows.append(['Weighted Avg', f"{report['weighted avg']['precision']:.3f}", f"{report['weighted avg']['recall']:.3f}", f"{report['weighted avg']['f1-score']:.3f}", int(report['weighted avg']['support'])])

    fig, ax = plt.subplots(figsize=(12, len(rows)*0.6 + 2))
    ax.axis('off')
    col_labels = ['Class', 'Precision', 'Recall', 'F1 Score', 'Support']
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2.4)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_fontsize(15)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#0072B2')
        else:
            cell.set_facecolor('#F7FBFF' if key[0] % 2 == 1 else 'white')
            if key[1] == 0:
                cell.set_text_props(weight='bold')
    plt.title('Detailed Classification Report by Class', fontweight='bold', fontsize=18, pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_PATH, 'detailed_classification_report.png'), dpi=150, bbox_inches='tight')
    plt.show()