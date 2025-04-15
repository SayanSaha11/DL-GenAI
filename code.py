import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dir = "/kaggle/input/kermany2018/OCT2017 /train"
val_dir = "/kaggle/input/kermany2018/OCT2017 /val"
test_images_dir = "/kaggle/input/kermany2018/OCT2017 /test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # OCT is grayscale; ViT expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images).logits
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {100 * correct / total:.2f}%")


torch.save(model.state_dict(), "vit_retinal_oct.pth")
print("Model saved as vit_retinal_oct.pth")


def predict_image(image_path):
    image = Image.open(image_path).convert("L").resize((224, 224))
    image = image.convert("RGB")  # Convert grayscale to RGB
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(pixel_values=image_tensor).logits
        _, predicted = torch.max(output, 1)

    return train_dataset.classes[predicted.item()]

# Predict on test sample images
if os.path.exists(test_images_dir):
    print("\n Predictions on new OCT images:")
    for file in os.listdir(test_images_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(test_images_dir, file)
            pred = predict_image(path)
            print(f"{file}:  {pred}")
else:
    print(f" Test image directory not found: {test_images_dir}")
