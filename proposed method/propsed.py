import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create folder for saving outputs if not exists
if not os.path.exists('model_outputs'):
    os.makedirs('model_outputs')

# --------------- Dataset Class ---------------
class DecomposedImageDataset(Dataset):
    def __init__(self, image_path, label, transform=None):
        self.image_path = image_path
        self.label = label
        self.transform = transform

    def reflector(self, img):
        # Simple reflector: Adjust brightness and contrast
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    def gamma_correction(self, img, gamma_value):
        invGamma = 1.0 / gamma_value
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def log_correction(self, img, c=1.0):
        img_log = c * (np.log1p(img / 255.0))  # Use np.log1p for numerical stability
        img_log = np.uint8(np.clip(img_log * 255, 0, 255))
        return img_log

    def __getitem__(self, idx):
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fixed size
        img = cv2.resize(img, (128, 128))

        # Apply reflector preprocessing and save
        reflected_img = self.reflector(img)
        cv2.imwrite(f'model_outputs/reflected_image.png', cv2.cvtColor(reflected_img, cv2.COLOR_RGB2BGR))

        # Generate and save gamma corrected images
        gamma_values = [0.6, 0.8, 1.0]
        gamma_imgs = [self.gamma_correction(img, gamma) for gamma in gamma_values]
        for i, gamma_img in enumerate(gamma_imgs):
            cv2.imwrite(f'model_outputs/gamma_image_{gamma_values[i]}.png', cv2.cvtColor(gamma_img, cv2.COLOR_RGB2BGR))

        # Generate and save log corrected images
        log_coeffs = [0.1, 0.3, 0.5]
        log_imgs = [self.log_correction(img, c) for c in log_coeffs]
        for i, log_img in enumerate(log_imgs):
            cv2.imwrite(f'model_outputs/log_image_{log_coeffs[i]}.png', cv2.cvtColor(log_img, cv2.COLOR_RGB2BGR))

        # Combine all images for model input
        all_imgs = gamma_imgs + log_imgs
        all_imgs = [img / 255.0 for img in all_imgs]  # Normalize to [0,1]
        all_imgs = np.stack(all_imgs, axis=0)  # Shape: (6, 128, 128, 3)

        # Convert to torch tensor and rearrange dims to (6, 3, 128, 128)
        all_imgs = torch.tensor(all_imgs, dtype=torch.float32).permute(0, 3, 1, 2)

        # Reshape to combine channels: (6, 3, 128, 128) -> (18, 128, 128)
        all_imgs = all_imgs.reshape(18, 128, 128).contiguous()

        label = torch.tensor(self.label, dtype=torch.long)

        return all_imgs, label

    def __len__(self):
        return 1  # Single image with multiple transformations

# --------------- CNN Model ---------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)  # Input: 18 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # After two 2x2 pooling: 128/4 = 32
        self.fc2 = nn.Linear(256, 2)  # Binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------- Load a single image ---------------
image_path = 'test10.jpg'  # Replace with your image path
label = 1  # Example label (0 or 1 for binary classification)

# Initialize the dataset
dataset = DecomposedImageDataset(image_path, label)

# DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --------------- Train the Model ---------------
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images = images.to(device)  # Shape: (1, 18, 128, 128)
        labels = labels.to(device)  # Shape: (1,)

        optimizer.zero_grad()
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# --------------- Save the Model and Plot Losses ---------------
torch.save(model.state_dict(), 'model_outputs/cnn_model.pth')

plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.savefig('model_outputs/loss_plot.png')
plt.close()

# --------------- Evaluate the Model ---------------
model.eval()
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted: {predicted.item()}, Actual: {labels.item()}')

print("Training and evaluation completed. Model, loss plot, and transformed images saved in 'model_outputs'.")