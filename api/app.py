from fastapi import FastAPI
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

app = FastAPI()

classes_info = pd.DataFrame({
    'class': ['background', 'flood', 'sky', 'building'],
    'grayscale_value': [0, 255, 170, 85],
    'class_index': [0, 1, 2, 3]
})

model = smp.Unet(
    encoder_name='vgg19',
    encoder_weights=None,
    in_channels=3,
    classes=len(classes_info)
)
model.load_state_dict(torch.load('models/best_params.pt', weights_only=True))
model.eval()

IMAGE_SIZE = (448, 448)

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
model.to(device)

@app.get('/predict')
def predict():
    images_path = 'test_images/'
    predictions_path = 'test_predictions/'

    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        image = Image.open(image_path).convert('RGB')
        origin_image_height = image.height
        origin_image_width = image.width
        transformed_image = transforms(image)
        output = model(transformed_image.to(device).unsqueeze(0)).squeeze()
        mask = output.argmax(dim=0)
        resize = v2.Resize(
            (origin_image_height, origin_image_width),
            interpolation=cv2.INTER_NEAREST
        )
        origin_size_mask = resize(mask.unsqueeze(0)).squeeze()
        colored_mask = index_to_color(origin_size_mask)
        mask_name = image_name.split('.')[0] + '.png'
        plt.imsave(os.path.join(predictions_path, mask_name), colored_mask)

def index_to_color(mask):
    mask_colors = []

    for index in range(len(classes_info)):
        color = classes_info[classes_info['class_index'] == index]['grayscale_value'].item()
        mask_colors.append(torch.where(mask == index, color, 0).tolist())

    return torch.tensor(mask_colors).max(dim=0).values