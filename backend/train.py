import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
import os

# Allow Pillow to handle truncated/corrupted image bytes automatically
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    # Update to "Dataset" to match your exact directory casing
    DATA_DIR = "Dataset" if os.path.exists("Dataset") else "dataset"
    BATCH_SIZE = 32
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print(f"Loading data from: {DATA_DIR}")

    # Image transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets using folder structure
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory '{train_dir}' not found. Please verify folder placement.")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Class Mapping detected: {train_dataset.class_to_idx}")

    # Load Pretrained DenseNet121 & Freeze Base Layers
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, 1))

    # Identify numerical index mapped to 'fractured'
    fractured_idx = train_dataset.class_to_idx.get("fractured", 0)

    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    print("\n--- Starting Fine-Tuning ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Explicitly map target tensor so 'fractured' is 1.0 and 'not fractured' is 0.0
            binary_labels = (labels == fractured_idx).float().unsqueeze(1).to(DEVICE)
            images = images.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                binary_labels = (labels == fractured_idx).float().to(DEVICE)
                images = images.to(DEVICE)
                
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).squeeze().float()
                val_corrects += torch.sum(preds == binary_labels)

        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Save fine-tuned output weights
    output_path = "fracture_densenet.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining Complete! Fine-tuned weights saved to '{output_path}'.")

if __name__ == "__main__":
    main()