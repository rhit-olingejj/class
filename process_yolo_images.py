import os

import kagglehub
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

# Define paths and parameters
path = r"yoloValidationDataset"
#path = kagglehub.dataset_download("lyensoetanto/vehicle-images-dataset")
target_size = (224, 224)
batch_size = 32

# Load class labels
class_to_idx = {}
data_list = []
labels_list = []

# Define 3 Layer Neural Network
class VehicleNet3Layer(nn.Module):
    def __init__(self, num_classes):
        super(VehicleNet3Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 125)
        self.bn1 = nn.BatchNorm1d(125)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(125, 75)
        self.bn2 = nn.BatchNorm1d(75)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(75, 25)
        self.bn3 = nn.BatchNorm1d(25)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(25, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# Define 5 Layer Neural Network
class VehicleNet5Layer(nn.Module):
    def __init__(self, num_classes):
        super(VehicleNet5Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 125)
        self.bn1 = nn.BatchNorm1d(125)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.05)

        self.fc2 = nn.Linear(125, 75)
        self.bn2 = nn.BatchNorm1d(75)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.01)

        self.fc3 = nn.Linear(75, 25)
        self.bn3 = nn.BatchNorm1d(25)
        self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(0.01)

        self.fc4 = nn.Linear(25, 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.relu4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(20, 15)
        self.bn5 = nn.BatchNorm1d(15)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(15, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.relu5(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x


# Define 7 Layer Neural Network
class VehicleNet7Layer(nn.Module):
    def __init__(self, num_classes):
        super(VehicleNet7Layer, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 125)
        self.bn1 = nn.BatchNorm1d(125)
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.05)

        self.fc2 = nn.Linear(125, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.01)

        self.fc3 = nn.Linear(100, 75)
        self.bn3 = nn.BatchNorm1d(75)
        self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(0.01)

        self.fc4 = nn.Linear(75, 50)
        self.bn4 = nn.BatchNorm1d(50)
        self.relu4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(50, 25)
        self.bn5 = nn.BatchNorm1d(25)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(25, 20)
        self.bn6 = nn.BatchNorm1d(20)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(20, 20)
        self.bn7 = nn.BatchNorm1d(20)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        # x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        # x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.relu4(self.bn4(self.fc4(x)))
        x = self.relu5(self.bn5(self.fc5(x)))
        x = self.relu6(self.bn6(self.fc6(x)))
        x = self.relu7(self.bn7(self.fc7(x)))
        x = self.fc8(x)
        return x

def preprocess_image(image_path):
    """Preprocess an image for model inference."""
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1)
    return img_tensor


for idx, class_name in enumerate(os.listdir(path)):
    class_to_idx[class_name] = idx
    class_dir = os.path.join(path, class_name)

    for filename in os.listdir(class_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_dir, filename)
            data_list.append(preprocess_image(img_path))
            labels_list.append(idx)

# Convert validation data into tensors
data_tensor = torch.stack(data_list)
labels_tensor = torch.tensor(labels_list, dtype=torch.long)

# Create validation DataLoader
val_dataset = data.TensorDataset(data_tensor, labels_tensor)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_to_idx)


# Use the model from the other files.
# VehicleNet3Layer - 3 layer model from final.py
# VehicleNet5Layer - 5 layer model from final_five_layer.py
# VehicleNet7Layer - 7 layer model from seven-layer-model.py
model = VehicleNet3Layer(num_classes).to(device)


# Load the saved model
model_path = "vehicle_classification_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model from {model_path}")

# Function to evaluate the model
def evaluate_model(loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_to_idx.keys(), yticklabels=class_to_idx.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Validation Confusion Matrix 7-Layer (Accuracy: {acc:.2f})")
    plt.savefig("validation_confusion_matrix.png")
    print(f"Validation confusion matrix saved as validation_confusion_matrix.png")
    plt.close()

    print(f"Validation Accuracy: {acc:.2f}")

# Evaluate the model on validation dataset
evaluate_model(val_loader)
