import os
import warnings
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp

# Ignore warnings
warnings.filterwarnings('ignore')

# Paths to images and models
image_path = "input/0901.PNG"
save_root = "save_masks"

# Model files and corresponding save directories
models_info = {
    "efficientnet-b3": {
        "pt_file": "models/efficientnet-b3_SelfONN_FPN.pt",
        "save_folder": os.path.join(save_root, "masks_b3"),
    },
    "efficientnet-b4": {
        "pt_file": "models/efficientnet-b4_SelfONN_FPN.pt",
        "save_folder": os.path.join(save_root, "masks_b4"),
    },
    "efficientnet-b5": {
        "pt_file": "models/efficientnet-b5_SelfONN_FPN.pt",
        "save_folder": os.path.join(save_root, "masks_b5"),
    }
}

# Ensure all save directories exist
for model_name, info in models_info.items():
    os.makedirs(info["save_folder"], exist_ok=True)


image = Image.open(image_path).convert('L')  # Ensure grayscale mode


gray_image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40936773], std=[0.33767385])
])


input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 1, 512, 512]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)


for model_name, info in models_info.items():
    print(f"Processing {model_name}...")

    # Load checkpoint
    checkpoint = torch.load(info["pt_file"], map_location=device)

    # Define the model architecture correctly
    # model = smp.FPN(encoder_name=model_name, classes=1, activation=None)
    model = checkpoint['model']
    # Load model weights
    # if 'model' in checkpoint:
    #     model.load_state_dict(checkpoint['model'])
    # else:
    #     model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()  

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)  
        output_cpu = torch.sigmoid(output).squeeze().cpu().numpy()  

    threshold = 0.5
    binary_mask = (output_cpu > threshold).astype(np.uint8) * 255  


    binary_mask_img = Image.fromarray(binary_mask) 

    img_name = os.path.basename(image_path).split('.')[0]

    save_path = os.path.join(info["save_folder"], f"{img_name}_mask.png")
    binary_mask_img.save(save_path)

    print(f"Saved mask for {model_name} at: {save_path}")

print( "All masks saved successfully!")
