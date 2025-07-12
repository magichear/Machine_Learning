import torch
import argparse
import cv2
import os

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from config import ALL_CLASSES
from model import UNet

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir')
parser.add_argument(
    '--model',
    default='../outputs/model.pth',
    help='path to the model checkpoint'
)
parser.add_argument(
    '--imgsz',
    default=None,
    help='image resize resolution',
)
args = parser.parse_args()

out_dir = os.path.join('..', 'outputs', 'inference_results')
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(num_classes=len(ALL_CLASSES))
ckpt = torch.load(args.model, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    print(f"Image {i+1}")
    # Read the image.
    image = cv2.imread(os.path.join(args.input, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args.imgsz is not None:
        image = cv2.resize(image, (int(args.imgsz), int(args.imgsz)))

    image_copy = image.copy()
    image_copy = image_copy / 255.0
    image_tensor = torch.permute(
        torch.tensor(image_copy, dtype=torch.float32), (2, 0, 1)
    )
    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image_tensor, model, device)
    outputs = outputs
    segmented_image = draw_segmentation_map(outputs)
    
    final_image = image_overlay(image, segmented_image)
    cv2.imshow('Segmented image', final_image)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(out_dir, image_path), final_image)