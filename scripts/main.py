import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Object type map
object_type_map = {
    0: "Unknown",
    1: "Person",
    2: "Car",
    3: "Other Vehicle",
    4: "Other Object",
    5: "Bike"
}

# --- Custom Dataset Class ---
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.base_dir = images_dir
        self.transforms = transforms
        self.target_size = (800, 800)
        
        # Get all pair folders
        self.pair_folders = sorted([f for f in os.listdir(self.base_dir) 
                                  if f.startswith("Pair_") and 
                                  os.path.isdir(os.path.join(self.base_dir, f))])
        
        print(f"Found {len(self.pair_folders)} pair folders")  # Debug print
        
        # Create list of all valid image pairs
        self.image_pairs = []
        for folder in self.pair_folders:
            folder_path = os.path.join(self.base_dir, folder)
            # Extract base name from folder
            base_name = folder.replace("Pair_", "")
            x, y = base_name.split('_')[-2:]
            base_name = base_name.replace(f"_{x}_{y}", "")

            # Construct the two timestamps we expect
            
            first_timestamp = base_name + f"_{x}"
            second_timestamp = base_name + f"_{y}"
            
            # Construct full file paths
            img1_path = os.path.join(folder_path, f"{first_timestamp}.png")
            img2_path = os.path.join(folder_path, f"{second_timestamp}.png")
            annot2_path = os.path.join(folder_path, f"{second_timestamp}.annotation.txt")
                        
            # Check if all required files exist
            if os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(annot2_path):
                self.image_pairs.append((img1_path, img2_path, annot2_path))
            else:
                continue
        
        print(f"\nTotal valid pairs found: {len(self.image_pairs)}")
        
        if len(self.image_pairs) == 0:
            raise ValueError("No valid image pairs found. Please check the data directory structure and file naming.")

    def __getitem__(self, idx):
        # Get paths for the pair
        img1_path, img2_path, annot_path = self.image_pairs[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Resize and convert images to tensors
        img1 = torchvision.transforms.Resize(self.target_size)(img1)
        img2 = torchvision.transforms.Resize(self.target_size)(img2)

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        else:
            img1 = torchvision.transforms.ToTensor()(img1)
            img2 = torchvision.transforms.ToTensor()(img2)

        # Compute pixel-wise difference
        diff_img = torch.abs(img2 - img1)

        # Load annotations for the second image
        boxes, labels = [], []
        with open(annot_path, 'r') as f:
            for line in f:
                data = list(map(int, line.split()))
                if len(data) == 8 and data[7] in object_type_map:
                    boxes.append([data[3], data[4], data[3] + data[5], data[4] + data[6]])
                    labels.append(data[7])

        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return diff_img, target

    def __len__(self):
        return len(self.image_pairs)

# --- Model Builder ---
def build_model(num_classes):
    model = FasterRCNN(backbone=torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=True),
                       num_classes=num_classes)
    return model

# --- Save and Load Functions ---
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))

# --- Evaluate and Find Optimal Threshold ---
def calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    # Convert inputs to correct tensor format if they aren't already
    if len(pred_boxes) == 0:
        return 0, 0, len(true_boxes)
    
    pred_boxes = pred_boxes.to(true_boxes.device)
    
    # IoU-based metric calculation
    tp, fp, fn = 0, 0, len(true_boxes)

    # Calculate IoU between all pred and true boxes at once
    iou_matrix = torchvision.ops.box_iou(pred_boxes, true_boxes)
    
    for i, (iou_row, pl) in enumerate(zip(iou_matrix, pred_labels)):
        if len(true_boxes) == 0:
            fp += 1
            continue
            
        max_iou, max_idx = iou_row.max(0)
        if max_iou > iou_threshold and pl in true_labels:
            tp += 1
            fn -= 1
        else:
            fp += 1
    
    return tp, fp, fn

def find_optimal_threshold(model, val_dataloader, device):
    model.eval()  # Set model to evaluation mode
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold, best_f1 = 0.5, 0

    with torch.no_grad():
        for threshold in thresholds:
            total_tp, total_fp, total_fn = 0, 0, 0

            for images, targets in val_dataloader:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for output, target in zip(outputs, targets):
                    # Move tensors to the same device
                    pred_boxes = output['boxes'][output['scores'] > threshold].to(device)
                    pred_labels = output['labels'][output['scores'] > threshold].to(device)
                    true_boxes = target['boxes'].to(device)
                    true_labels = target['labels'].to(device)

                    tp, fp, fn = calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

            precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
            recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"Optimal Threshold: {best_threshold:.2f}, Best F1-Score: {best_f1:.4f}")
    return best_threshold

# --- Training Function ---
def train_model(model, dataloader, optimizer, device, epochs=10, save_path="model_weights.pth"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")
    save_model(model, save_path)

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up base directory
    base_dir = "../data"
    pair_folders = [f for f in os.listdir(base_dir) if f.startswith("Pair_") and os.path.isdir(os.path.join(base_dir, f))]
    
    # Create lists to store all image paths and annotation paths
    all_image_paths = []
    all_annotation_paths = []
    
    # Collect all image and annotation paths
    for folder in pair_folders:
        folder_path = os.path.join(base_dir, folder)
        # Sort to ensure consistent ordering
        images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
        annotations = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        
        for img in images:
            all_image_paths.append(os.path.join(folder_path, img))
        for ann in annotations:
            all_annotation_paths.append(os.path.join(folder_path, ann))

    # Update the dataset initialization
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Modify the CustomObjectDetectionDataset class initialization
    dataset = CustomObjectDetectionDataset(
        images_dir=base_dir,  # Pass the base directory
        annotations_dir=base_dir,  # Pass the base directory
        transforms=transform
    )

    # Rest of the code remains the same
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = build_model(num_classes=len(object_type_map) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    weights_path = "fasterrcnn_weights.pth"
    if os.path.exists(weights_path):
        print("Loading saved model weights...")
        load_model(model, weights_path, device)
    else:
        print("Training model...")
        train_model(model, train_dataloader, optimizer, device, epochs=10, save_path=weights_path)

    print("Evaluating to find optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, val_dataloader, device)

    print(f"Optimal threshold determined: {optimal_threshold}")

    
