# inference.py
import torch
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# आधी



# Grad-CAM import
from grad_cam import GradCAM
from grad_cam.utils.model_targets import ClassifierOutputTarget
from grad_cam.utils.image import show_cam_on_image
# ----------------------------
# CONFIGURATION
# ----------------------------
IMAGE_PATH = "test_leaf.jpg"           # test image path
CHECKPOINT_PATH = "best_plant_disease_model.pth"
NUM_CLASSES = 10                       # change to 38 if your checkpoint has 38 classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# MODEL LOAD
# ----------------------------
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Option: If checkpoint and model class count mismatch, ignore head
state_dict = {k: v for k, v in checkpoint.items() if "head" not in k}
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ----------------------------
# PREDICTION
# ----------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = outputs.argmax(dim=1).item()
    confidence = probs[0][pred_class].item()

print(f"Predicted Class: {pred_class}, Confidence: {confidence:.2f}")

# ----------------------------
# GRAD-CAM HEATMAP
# ----------------------------
# For ViT, usually use last block norm
target_layers = [model.blocks[-1].norm1]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(DEVICE=="cuda"))
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
grayscale_cam = grayscale_cam[0, :]

# Overlay on image
rgb_img = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Display
plt.figure(figsize=(6,6))
plt.imshow(visualization)
plt.title(f"Prediction: {pred_class} (conf: {confidence:.2f})")
plt.axis("off")
plt.show()

# Optional: save heatmap
plt.imsave("heatmap_output.png", visualization)
