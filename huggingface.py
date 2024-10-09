import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Collate function to handle padding of ground truth bounding boxes
def collate_fn(batch):
    images, boxes = zip(*batch)
    
    # Convert list of images to tensor
    images = torch.stack(images)
    
    # Determine the maximum number of bounding boxes in the batch
    max_boxes = max([b.size(0) for b in boxes])
    
    # Pad all bounding boxes to have the same number of entries
    padded_boxes = []
    for b in boxes:
        num_boxes = b.size(0)
        padding = torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32)
        padded_boxes.append(torch.cat([b, padding], dim=0))
    
    # Stack all padded boxes
    boxes = torch.stack(padded_boxes)
    
    return images, boxes


# Function to visualize multiple bounding boxes
def visualize_bboxes(image_tensor, bboxes):
    image = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)  # Denormalize the image

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        image_with_bbox = cv2.rectangle(image.copy(), (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

    plt.imshow(image_with_bbox)
    plt.axis('off')
    plt.show()


# Custom Dataset for images and bounding boxes
class EquationImageDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, new_size=(224, 224)):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.new_size = new_size  # Target image size (224x224)
        self.image_filenames = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(image_path).convert('RGB')

        # Parse XML annotation and adjust bounding boxes
        annotation_path = os.path.join(self.annotation_dir, self.image_filenames[idx].replace('.png', '.xml').replace('.jpg', '.xml'))
        boxes = self.parse_xml(annotation_path, img.size)  # Pass the original image size

        if self.transform:
            img = self.transform(img)

        return img, torch.as_tensor(boxes, dtype=torch.float32)

    def parse_xml(self, annotation_path, original_size):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        original_width, original_height = original_size  # Original image dimensions

        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Normalize bounding box to the new size (224x224)
            xmin = xmin * self.new_size[0] / original_width
            ymin = ymin * self.new_size[1] / original_height
            xmax = xmax * self.new_size[0] / original_width
            ymax = ymax * self.new_size[1] / original_height

            boxes.append([xmin, ymin, xmax, ymax])

        return boxes


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Update the paths to the image directory and XML annotations directory
image_dir = 'testFile1'  # Images without circles
annotation_dir = 'testFile2'  # XML annotations

# Create the Dataset and DataLoader
dataset = EquationImageDataset(image_dir=image_dir, annotation_dir=annotation_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Updated loss function to handle padding entries and mask them correctly
def multi_box_loss(predicted_bboxes, ground_truth_bboxes, device):
    # Create a mask for valid bounding boxes
    mask = (ground_truth_bboxes.sum(dim=2) != 0).unsqueeze(2).to(device)  # Shape: (batch_size, num_boxes, 1)
    
    # Ensure predicted boxes are on the same device and have the same shape
    predicted_bboxes = predicted_bboxes.to(device)
    
    # Apply the mask to only consider non-padded boxes in the loss
    masked_predicted = predicted_bboxes * mask
    masked_ground_truth = ground_truth_bboxes.to(device) * mask
    
    # Compute the loss only for valid bounding boxes
    loss = nn.MSELoss()(masked_predicted, masked_ground_truth)
    
    return loss


class ViTWithMultiBoxRegression(nn.Module):
    def __init__(self, model, num_boxes=4):
        super(ViTWithMultiBoxRegression, self).__init__()
        self.vit = model
        self.num_boxes = num_boxes
        self.regression_head = nn.Linear(model.config.hidden_size, num_boxes * 4)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        bbox_coords = self.regression_head(outputs.pooler_output)
        return bbox_coords.view(-1, self.num_boxes, 4)


# Load the pre-trained ViT model and add a regression head
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model_with_regression = ViTWithMultiBoxRegression(model, num_boxes=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_with_regression.to(device)

optimizer = optim.AdamW(model_with_regression.parameters(), lr=1e-4)

# Load the saved model weights (ignore size mismatches for the regression head)
def load_pretrained_vit_with_custom_head(model, model_path):
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # Filter out the layers that don't match the current model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


model_path = 'model_with_regression.pth'
if os.path.exists(model_path):
    print("Loading previously trained model...")
    load_pretrained_vit_with_custom_head(model_with_regression, model_path)


# Training loop
num_epochs = 2000
num_boxes = 4

for epoch in range(num_epochs):
    model_with_regression.train()
    running_loss = 0.0

    for batch in dataloader:
        img_no_circle, ground_truth_bboxes = batch
        img_no_circle = img_no_circle.to(device)
        ground_truth_bboxes = ground_truth_bboxes.to(device)

        optimizer.zero_grad()
        predicted_bboxes = model_with_regression(img_no_circle)

        # Calculate the loss
        loss = multi_box_loss(predicted_bboxes, ground_truth_bboxes, device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")


    if epoch == num_epochs - 1:
        torch.save(model_with_regression.state_dict(), model_path)
        print("Model saved.")
        
        # Visualize the prediction
        model_with_regression.eval()
        with torch.no_grad():
            sample_img_no_circle, _ = next(iter(dataloader))
            sample_img_no_circle = sample_img_no_circle.to(device)
        
            # Get the predicted bounding boxes for a single image
            predicted_bboxes = model_with_regression(sample_img_no_circle[0:1]).cpu().numpy().reshape(-1, 4)

        # Visualize the prediction on the first image of the batch
        visualize_bboxes(sample_img_no_circle[0], predicted_bboxes)
