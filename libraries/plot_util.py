import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from constants import Base


def save_fig(image_path):
    plt.savefig(image_path, format='jpeg')
    plt.close()


def plot_PCA(X, y, plot_image_file):
    # Flatten X to 2D array
    X_flat = X.reshape(X.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    # Create scatter plot of X_pca with color-coded labels
    plt.figure(figsize=(8, 6))
    for class_label in np.unique(y):
        indices = np.where(y == class_label)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Class {class_label}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: X vs y')

    plt.legend()
    plt.tight_layout()
    save_fig(plot_image_file)


def plot_distribution(X, y, plot_image_file):
    # Create subplots for each class
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
    fig.suptitle("Distribution of X_train across Classifications", fontsize=16, y=0.92)

    # Select the class labels for plotting
    class_labels = [0, 1, 2]

    # Iterate over the class labels and plot the distribution of X_train for each class
    for i, class_label in enumerate(class_labels):
        # Select the samples for the current class
        samples = X[y == class_label]

        # Flatten the samples for plotting
        samples_flat = samples.flatten()

        # Create a list of labels for the box plot
        labels = [str(class_label)]

        # Create a list of all samples from other classes
        other_samples = X[y != class_label]
        other_samples_flat = other_samples.flatten()

        # Append labels for other classes
        labels.append("Others")

        # Combine the current class samples and samples from other classes
        data = [samples_flat, other_samples_flat]

        # Plot the box plot
        ax = axes[i]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"Class {class_label}")
        ax.set_ylabel("Value")

    plt.tight_layout()
    save_fig(plot_image_file)


def plot_confusion_matrix(predicted_labels, true_labels):
    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    # Define class labels
    class_labels = ['Class 0', 'Class 1', 'Class 2']
    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = torch.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    # Fill the matrix cells with the values
    thresh = cm.max() / 2.0
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    image_name = "confusion_mat.jpeg"

    image_path = os.path.join(Base.BASE_IMAGE_PATH, image_name)
    plt.savefig(image_path, format="jpeg")
    plt.close()
