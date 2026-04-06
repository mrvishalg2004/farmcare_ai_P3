# models_helper.py
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import re

# -------------------------
# === Helper: Preprocess ===
# -------------------------
def preprocess_image(pil_img: Image.Image, meta: dict):
    """
    Resize/normalize based on meta. meta can contain:
    - input_size: (H,W) default (224,224)
    - mean, std: normalization
    """
    input_size = meta.get("input_size", (224,224))
    mean = meta.get("mean", [0.485,0.456,0.406])
    std  = meta.get("std", [0.229,0.224,0.225])
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform(pil_img).unsqueeze(0)  # [1,C,H,W]

# -------------------------
# === Load Model / Meta ===
# -------------------------
def load_model_and_meta(path):
    """
    Tries to load model. Supports:
    1) Whole model saved via torch.save(model)
    2) state_dict: then user must provide model class in `custom_model.py` and set meta['state_dict_only']=True
       For convenience, if state_dict-only is detected, we look for `YourModel()` defined in this file.
    Returns: model, meta(dict)
    IMPORTANT: You may need to edit this function to import your custom model class.
    """
    meta = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(path, map_location=device)
    # If it's a dict with multiple keys, try to detect saved meta
    if isinstance(data, dict):
        # Common pattern: saved = {'model_state_dict':..., 'labels':..., 'meta': ...}
        if 'model_state_dict' in data or 'state_dict' in data:
            state = data.get('model_state_dict', data.get('state_dict'))
            # Try to import user model: replace below with your model class if needed
            try:
                # Attempt to import common torchvision models by name
                arch = data.get('arch', None)
                if arch is not None:
                    # user may have stored arch name
                    if "resnet" in arch.lower():
                        from torchvision.models import resnet18
                        model = resnet18(num_classes=data.get("num_classes",1000))
                    else:
                        # fallback: user must edit to provide model class
                        raise Exception("Unknown arch in metadata. Edit load_model_and_meta to construct right model.")
                else:
                    # fallback - assume resnet18 with fitted output classes (user should change)
                    from torchvision.models import resnet18
                    model = resnet18(num_classes=data.get("num_classes",1000))
                model.load_state_dict(state)
                model.eval().to(device)
                meta['arch_type'] = 'cnn'
                # propagate labels if present
                if 'labels' in data:
                    meta['labels'] = data['labels']
                meta['input_size'] = data.get('input_size', (224,224))
                meta['mean'] = data.get('mean', [0.485,0.456,0.406])
                meta['std']  = data.get('std', [0.229,0.224,0.225])
                return model, meta
            except Exception as e:
                raise RuntimeError(f"State-dict loaded but couldn't construct model automatically: {e}\nEdit load_model_and_meta to import your model class and construct it, or save whole model with torch.save(model).")
        else:
            # Possibly entire model object or contains 'model' key
            # Try many fallbacks
            if 'model' in data and hasattr(data['model'], 'eval'):
                model = data['model']
                model.eval().to(device)
                meta['arch_type'] = detect_arch_type(model)
                if 'labels' in data:
                    meta['labels'] = data['labels']
                return model, meta
            # maybe entire model object
    # If loading raw object (torch.save(model))
    try:
        model = data
        model.eval().to(device)
        meta['arch_type'] = detect_arch_type(model)
        return model, meta
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}\nMake sure you saved the full model or provide model class to load state_dict.")

def detect_arch_type(model):
    """
    Heuristic to detect if model is CNN or ViT by checking attribute names.
    """
    s = str(model.__class__).lower()
    if 'vit' in s or 'visiontransformer' in s or 'transformer' in s:
        return 'vit'
    # check attr for blocks/attn
    if hasattr(model, 'blocks') or hasattr(model, 'attn'):
        return 'vit'
    # look for conv layers
    for _name, _module in model.named_modules():
        if _module.__class__.__name__.lower().startswith('conv'):
            return 'cnn'
    return 'unknown'

# -------------------------
# === Prediction util ===
# -------------------------
def predict(model, input_tensor, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
        pred_idx = int(np.argmax(probs))
        return logits.cpu().squeeze(0), probs, pred_idx

# -------------------------
# === Grad-CAM (CNN) ===
# -------------------------
class FeatureExtractor:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.target_layer = None
        # register hook to first matching layer name
        for name, module in model.named_modules():
            if name.endswith(target_layer_name) or (target_layer_name in name):
                self.target_layer = module
                break
        if self.target_layer is None:
            # fallback: find last conv
            convs = [(n,m) for n,m in model.named_modules() if m.__class__.__name__.lower().startswith('conv')]
            if convs:
                name, mod = convs[-1]
                self.target_layer = mod
                target_layer_name = name
            else:
                raise RuntimeError("No conv layer found for Grad-CAM. Edit FeatureExtractor target layer.")
        # hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

def gradcam_for_cnn(model, input_tensor, target_class=None, target_layer_name=None, device=None):
    """
    Basic Grad-CAM:
    - target_layer_name: string to match last conv, e.g. "layer4" for ResNet. If None, it picks last conv.
    Returns heatmap (H,W) normalized 0-1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # guess target layer name for common models
    if target_layer_name is None:
        guess = "layer4"  # resnet
        target_layer_name = guess

    extractor = FeatureExtractor(model, target_layer_name)
    input_tensor = input_tensor.to(device)
    # forward
    logits = model(input_tensor)
    if isinstance(logits, tuple):
        logits = logits[0]
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward(retain_graph=True)

    gradients = extractor.gradients.detach().cpu().numpy()[0]  # [C,H,W]
    activations = extractor.activation.detach().cpu().numpy()[0]  # [C,H,W]

    weights = np.mean(gradients, axis=(1,2))  # [C]
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # H,W
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    cam = cv2_resize_uint8(cam)
    return cam  # float 0-1

def cv2_resize_uint8(cam):
    import cv2
    # ensure 0-1 -> uint8
    cam_u8 = np.uint8(255 * cam)
    # small safety: if single pixel, expand
    return cam_u8

# -------------------------
# === ViT: Attention Rollout ===
# -------------------------
def vit_attention_rollout(model, input_tensor, device=None, head_fusion="mean", discard_ratio=0.9):
    """
    Approximate attention rollout for ViT-like models.
    This implementation expects that model has attribute `blocks` and each block has `attn` with `get_attn` or `qkv`.
    If your ViT model differs, edit this function to extract attention weights.
    Returns attention map resized to patch-grid (HxW) as float 0-1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # forward with hooks: we will try to collect attn matrices
    attn_weights = []

    # Try common timm-like structure
    for name, module in model.named_modules():
        # look for modules with 'attn' in name and attribute 'get_attention'
        if name.endswith("attn") or "attn" in name:
            # attempt to call module with return of attn
            # We can't easily get intermediate attn without modifying model; try typical patterns
            pass

    # fallback: try to run a forward that returns attentions if model supports it (timm ViT variant)
    try:
        out = model(input_tensor, return_attention=True)
        # expected out: (logits, attns) or dict
        if isinstance(out, tuple) and len(out) >= 2:
            attn_weights = out[1]  # list of [B,heads, tokens, tokens]
    except Exception:
        pass

    # If we couldn't get attn via return_attention, try to access blocks if available
    if not attn_weights:
        # Try to re-run forward while capturing qkv via hooks (best-effort, not guaranteed)
        attn_weights = []
        def hook_fn(module, inp, out):
            # Many ViT implementations put attn weights in out[1] or module.attn.attn if precomputed
            if isinstance(out, tuple) and len(out) >= 2:
                attn = out[1]
            elif hasattr(module, "attn_drop"):
                # fallback - skip
                attn = None
            else:
                attn = None
            if attn is not None:
                attn_weights.append(attn.detach().cpu().numpy())

        hooks = []
        for n, m in model.named_modules():
            if 'attn' in n:
                try:
                    hooks.append(m.register_forward_hook(hook_fn))
                except Exception:
                    pass
        _ = model(input_tensor)
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    if not attn_weights:
        raise RuntimeError("Couldn't extract attention weights from ViT model automatically. Edit `vit_attention_rollout` to match your ViT implementation.")

    # attn_weights is list of arrays [B, heads, tokens, tokens]. We'll compute rollout.
    # convert to numpy
    attn = np.array([a[0] for a in attn_weights])  # [layers, heads, tokens, tokens] maybe
    # average heads
    if attn.ndim == 4:
        attn_heads_fused = attn.mean(axis=1)  # [layers, tokens, tokens]
    else:
        attn_heads_fused = attn  # best-effort

    # add identity and multiply
    num_layers = attn_heads_fused.shape[0]
    rollout = np.eye(attn_heads_fused.shape[-1])
    for i in range(num_layers):
        a = attn_heads_fused[i]
        # optionally zero low attention weights
        flat = a.flatten()
        # normalize
        a = a / (a.sum(-1, keepdims=True) + 1e-9)
        rollout = a @ rollout

    # Extract [CLS] token attention to patches (assume cls is index 0)
    cls_attn = rollout[0, 1:]  # skip cls->cls
    # get spatial map size: assume square
    size = int(np.sqrt(cls_attn.shape[0]))
    if size*size != cls_attn.shape[0]:
        # fallback: just reshape as 1 x N
        attn_map = cls_attn.reshape(1, -1)
    else:
        attn_map = cls_attn.reshape(size, size)
    # normalize 0-1
    attn_map = attn_map - attn_map.min()
    if attn_map.max() != 0:
        attn_map = attn_map / attn_map.max()
    return attn_map
