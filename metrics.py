import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """Calculates and prints classification metrics."""
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return report, conf_matrix
