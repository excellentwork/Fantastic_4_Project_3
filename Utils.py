# utils.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import models as keras_models

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

# Function to load and summarize a model
def load_and_summarize_model(model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Print the model summary
    model.summary()
    
    return model


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models as keras_models

def visualize_feature_maps(model, X_input, max_layers=8, max_features=10):
    # Create a model that will return these outputs, given the model input
    layer_outputs = [layer.output for layer in model.layers[:max_layers]]  # Extract outputs of the first `max_layers` layers
    activation_model = keras_models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(X_input)
    
    # Print shapes of activations for debugging
    for i, layer_activation in enumerate(activations):
        print(f"Layer {i} activation shape: {layer_activation.shape}")
    
    # Plot the feature maps
    for layer_activation in activations:
        # Check if the activation is 4D (batch_size, height, width, channels)
        if len(layer_activation.shape) == 4:
            n_features = min(layer_activation.shape[-1], max_features)  # Limit the number of features
            size = layer_activation.shape[1]
            display_grid = np.zeros((size, size * n_features))
            
            for i in range(n_features):
                x = layer_activation[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = x
                
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
        else:
            print(f"Skipping layer with shape {layer_activation.shape} (not 4D)")


# Class Activation Maps
def plot_cam(model, img_array, class_idx, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    plt.matshow(heatmap)
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)

    plt.figure()
    plt.plot(recall, precision, lw=2, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc_curve(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_conv_filters(layer, layer_name):
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='viridis')
            ix += 1
    plt.show()

def plot_misclassified_examples(X_test, y_true, y_pred, class_names, num_examples=5):
    misclassified_indices = np.where(y_true != y_pred)[0]
    if len(misclassified_indices) < num_examples:
        num_examples = len(misclassified_indices)
    plt.figure(figsize=(10, 10))
    for i in range(num_examples):
        idx = misclassified_indices[i]
        plt.subplot(1, num_examples, i+1)
        plt.imshow(X_test[idx])
        plt.title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
        plt.axis('off')
    plt.show()
