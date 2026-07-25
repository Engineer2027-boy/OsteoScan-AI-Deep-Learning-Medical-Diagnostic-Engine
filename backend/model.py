import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import io
import base64

# ==========================================
# 1. CLAHE IMAGE PREPROCESSING
# ==========================================
def preprocess_clahe(image_bytes: bytes) -> Image.Image:
    # Decode image bytes to OpenCV grayscale array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image format or unreadable file")
    
    # Apply Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # Convert to 3-channel RGB for PyTorch vision models
    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb), img

# ==========================================
# 2. DENSENET-121 ARCHITECTURE
# ==========================================
class XRayFractureNet(nn.Module):
    def __init__(self, pretrained=True):
        super(XRayFractureNet, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1) # Binary logit output
        )

    def forward(self, x):
        return self.densenet(x)

# ==========================================
# 3. GRAD-CAM EXPLAINABILITY ENGINE
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        output.backward(gradient=torch.ones_like(output))
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        prob = torch.sigmoid(output).item()
        return cam, prob

# ==========================================
# 4. INFERENCE PIPELINE ENTRYPOINT
# ==========================================
class FracturePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XRayFractureNet(pretrained=True).to(self.device)
        self.model.eval()
        self.grad_cam = GradCAM(self.model, self.model.densenet.features.denseblock4)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes: bytes):
        pil_img, raw_cv_img = preprocess_clahe(image_bytes)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        heatmap, prob = self.grad_cam.generate(input_tensor)
        
        # Prepare heatmap overlay
        orig_resized = cv2.resize(raw_cv_img, (224, 224))
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        # Convert overlays to Base64 strings for React UI rendering
        _, overlay_buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(overlay_buf).decode('utf-8')
        
        return {
            "fracture_detected": prob > 0.5,
            "confidence": round(prob * 100, 2),
            "heatmap_base64": f"data:image/png;base64,{overlay_b64}"
        }
