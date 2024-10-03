import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTFeatureExtractor
import cv2
import numpy as np
def visualize_bbox(image_tensor, bbox):
    # Convert the tensor back to a NumPy array (for OpenCV)
    image = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)  # Denormalize the image

    # Draw the predicted bounding box on the image (in red)
    x_min, y_min, x_max, y_max = bbox
    image_with_bbox = cv2.rectangle(image.copy(), (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    # Display the image with the bounding box
    plt.imshow(image_with_bbox)
    plt.axis('off')  # Turn off axis labels
    plt.show()


# Custom Dataset for loading image pairs
class EquationImageDataset(Dataset):
    def __init__(self, image_dir_no_circle, image_dir_with_circle, transform=None):
        self.image_dir_no_circle = image_dir_no_circle
        self.image_dir_with_circle = image_dir_with_circle
        self.transform = transform
        self.image_pairs = [f for f in sorted(os.listdir(image_dir_no_circle)) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_no_circle_path = os.path.join(self.image_dir_no_circle, self.image_pairs[idx])
        img_with_circle_path = os.path.join(self.image_dir_with_circle, self.image_pairs[idx])

        img_no_circle = Image.open(img_no_circle_path).convert("RGB")
        img_with_circle = Image.open(img_with_circle_path).convert("RGB")

        if self.transform:
            img_no_circle = self.transform(img_no_circle)
            img_with_circle = self.transform(img_with_circle)

        return img_no_circle, img_with_circle

# Transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Paths to your images
image_dir_no_circle = '/Users/jackyzhang/ClassTranscribeOCR/testFile1'
image_dir_with_circle = '/Users/jackyzhang/ClassTranscribeOCR/testFile2'

# Create Dataset and DataLoader
dataset = EquationImageDataset(image_dir_no_circle, image_dir_with_circle, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Function to extract bounding boxes from the target image
def extract_bboxes_from_target(image_tensor):
    image = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    image = (image * 255).astype(np.uint8)  # Convert from normalized float back to uint8 format

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return torch.tensor([x, y, x + w, y + h], dtype=torch.float32, device=device)
    return torch.tensor([0, 0, 0, 0], dtype=torch.float32, device=device)

# Model with regression head for bounding box prediction
class ViTWithRegression(nn.Module):
    def __init__(self, model):
        super(ViTWithRegression, self).__init__()
        self.vit = model
        self.regression_head = nn.Linear(model.config.hidden_size, 4)  # 4 bounding box coordinates

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        bbox_coords = self.regression_head(outputs.pooler_output)  # Use pooled output
        return bbox_coords

# Load the pre-trained ViT model
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model_with_regression = ViTWithRegression(model)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_with_regression.to(device)

# Initialize optimizer and loss function
optimizer = optim.AdamW(model_with_regression.parameters(), lr=1e-4)
mse_loss = nn.MSELoss()

model_path = 'model_with_regression.pth'
if os.path.exists(model_path):
    print("Loading previously trained model...")
    model_with_regression.load_state_dict(torch.load(model_path))

# Training Loop
num_epochs = 100000
for epoch in range(num_epochs):
    model_with_regression.train()
    running_loss = 0.0
    
    for batch in dataloader:
        img_no_circle, img_with_circle = batch
        img_no_circle = img_no_circle.to(device)

        # Extract ground truth bounding boxes from the image with circles
        ground_truth_bboxes = torch.stack([extract_bboxes_from_target(img) for img in img_with_circle], dim=0)

        # Forward pass: get the predicted bounding boxes
        optimizer.zero_grad()
        predicted_bboxes = model_with_regression(img_no_circle)

        # Calculate loss between predicted and ground truth bounding boxes
        loss = mse_loss(predicted_bboxes, ground_truth_bboxes)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    if epoch == num_epochs - 1:
        torch.save(model_with_regression.state_dict(), model_path)
        print("Model saved.")
# After training, evaluate the model on test images to see how well it predicts bounding boxes
    #model_with_regression.eval()
    #with torch.no_grad():
        #sample_img_no_circle, _ = next(iter(dataloader))
        #sample_img_no_circle = sample_img_no_circle.to(device)
        
        # Get the predicted bounding box for a single image
        #predicted_bbox = model_with_regression(sample_img_no_circle[0:1]).cpu().numpy().flatten()

        # Visualize the prediction on the first image of the batch
        #visualize_bbox(sample_img_no_circle[0], predicted_bbox)