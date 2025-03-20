import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import kagglehub
from collections import Counter

# Define paths and parameters
# path = r"C:\Users\Jacob Olinger\.cache\kagglehub\datasets\lyensoetanto\vehicle-images-dataset\versions\1"
# path = kagglehub.dataset_download("lyensoetanto/vehicle-images-dataset")
# print(path)
path = r"/Users/nicholasscalzone/.cache/kagglehub/datasets/lyensoetanto/vehicle-images-dataset/versions/1"

num_layers = 5

target_size = (224, 224)
batch_size = 32
num_epochs = 50
learning_rate = 0.0005

data_list = []
labels_list = []
class_to_idx = {}

# Convert images into tensors
def preprocess_image(image_path):
    img = Image.open(image_path).resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1)
    return img_tensor

# Load images
for idx, class_name in enumerate(os.listdir(path)):
    class_to_idx[class_name] = idx
    class_dir = os.path.join(path, class_name)

    for filename in os.listdir(class_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_dir, filename)
            data_list.append(preprocess_image(img_path))
            labels_list.append(idx)

# Balance dataset
balanced_data, balanced_labels = [], []
min_samples = min([labels_list.count(c) for c in np.unique(labels_list)])

for class_label in np.unique(labels_list):
    class_indices = [i for i, label in enumerate(labels_list) if label == class_label]
    sampled_indices = np.random.choice(class_indices, min_samples, replace=False)

    for idx in sampled_indices:
        balanced_data.append(data_list[idx])
        balanced_labels.append(labels_list[idx])

# Convert to tensors
data_tensor = torch.stack(balanced_data)
labels_tensor = torch.tensor(balanced_labels, dtype=torch.long)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data_tensor, labels_tensor, test_size=0.2, stratify=labels_tensor, random_state=42
)

# Create DataLoaders
train_dataset = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = data.TensorDataset(X_val, y_val)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define Neural Network
class VehicleNet(nn.Module):
    def __init__(self, num_classes):
        super(VehicleNet, self).__init__()
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

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_to_idx)
model = VehicleNet(num_classes).to(device)

# Compute class weights
class_counts = Counter(y_train.numpy())
class_weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs

# Train the Model
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_acc = 100 * correct / total
    train_accuracies.append(train_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")
    scheduler.step()

# Save the model
model_path = "vehicle_classification_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Function to evaluate model and save confusion matrices
def evaluate_model(loader, dataset_type):
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
    plt.title(f"{num_layers} Layer Model {dataset_type} Confusion Matrix (Accuracy: {acc:.2f})")
    
    cm_filename = f"{num_layers}_layer_model_{dataset_type.lower()}_confusion_matrix_{num_epochs}_epochs.png"
    plt.savefig(cm_filename)
    print(f"{dataset_type} confusion matrix saved as {cm_filename}")
    plt.close()

# Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 7
# model = VehicleNet(num_classes).to(device)

# Load saved weights
# model.load_state_dict(torch.load("vehicle_classification_model.pth", map_location=device))
# model.eval()
# Evaluate on Training and Validation Data

evaluate_model(train_loader, "Train")
evaluate_model(val_loader, "Validation")

# Save Training Accuracy Plot
plt.figure(figsize=(10, 10))
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', linestyle='-', color='b', label="Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"{num_layers} Layer Model Training Accuracy Across Epochs")
plt.legend()
plt.grid()
plt.savefig(f"{num_layers}_layer_model_training_accuracy_{num_epochs}_epochs.png")
print(f"{num_layers} Layer Model Training accuracy chart saved as training_accuracy.png")
plt.close()