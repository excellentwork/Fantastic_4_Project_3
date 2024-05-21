# utils.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss graphs.
    
    Args:
    history: A Keras History object.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints out the loss and accuracy.
    
    Args:
    model: The trained model to evaluate.
    X_test: Test features.
    y_test: True test labels.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix using seaborn's heatmap.
    
    Args:
    y_true: Actual true labels.
    y_pred: Model's predictions.
    classes: Array of label names.
    title: Title of the plot.
    cmap: Color map of the heatmap.
    """
    cm = confusion_matrix(y_true, np.round(y_pred).astype(int))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def calculate_additional_metrics(y_true, y_pred, threshold=0.5, average='binary'):
    """
    Calculates and prints precision, recall, and F1-score for the predictions.
    
    Args:
    y_true: Actual true labels (expected as integers).
    y_pred: Model's predictions, expected as probabilities for binary classification.
    threshold: Threshold for converting probabilities to binary labels (default is 0.5).
    average: Averaging method for multiclass classification ('binary', 'micro', 'macro', 'weighted').
             Use 'binary' for binary classification unless handling a specific multiclass setup.
    """
    # Check if predictions are probabilities and convert to binary labels using the threshold
    if y_pred.ndim == 1 and np.max(y_pred) <= 1 and np.min(y_pred) >= 0:  # Probabilities check
        y_pred = (y_pred > threshold).astype(int)
    
    try:
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
    except Exception as e:
        print("Error calculating metrics:", e)

# Example usage:
# y_test = [0, 1, 0, 1]
# y_pred = [0.1, 0.9, 0.3, 0.76]  # Probabilities output from a model
# calculate_additional_metrics(y_test, y_pred)
