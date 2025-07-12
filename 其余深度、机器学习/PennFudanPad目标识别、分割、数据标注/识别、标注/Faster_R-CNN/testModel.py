import os
import time
import torch
from engine import get_detection_model, evaluate
from utils.dataset import PennFudanDataset
from utils.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from utils.utils import collate_fn

# 设置路径
root_path = "../PennFudanPed"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
os.environ["TORCH_HOME"] = "./models"

# 加载测试集
dataset = PennFudanDataset(root_path, transforms=Compose([ToTensor()]))
torch.manual_seed(42)
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
test_loader = DataLoader(
    dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
)

# 加载 Faster R-CNN 模型
fasterrcnn_model_None = get_detection_model(
    num_classes=2, is_rcnn_pretrained=False, is_mask_rcnn=False
)
state_dict_None = torch.load("../models/10.pth")  # 使用 torch.load 加载权重文件
fasterrcnn_model_None.load_state_dict(state_dict_None)  # 加载权重
fasterrcnn_model_None.to(device)

# 加载 Faster R-CNN 模型
fasterrcnn_model = get_detection_model(
    num_classes=2, is_rcnn_pretrained=True, is_mask_rcnn=False
)
state_dict_faster = torch.load("../models/f/10.pth")  # 使用 torch.load 加载权重文件
fasterrcnn_model.load_state_dict(state_dict_faster)  # 加载权重
fasterrcnn_model.to(device)

# 加载 Mask R-CNN 模型
maskrcnn_model = get_detection_model(
    num_classes=2, is_rcnn_pretrained=True, is_mask_rcnn=True
)
state_dict_mask = torch.load("../models/m/10.pth")  # 使用 torch.load 加载权重文件
maskrcnn_model.load_state_dict(state_dict_mask)  # 加载权重
maskrcnn_model.to(device)

# 定量分析
print("Evaluating Faster R-CNN_None...")
start_time = time.time()
fasterrcnn_none_map = evaluate(fasterrcnn_model_None, test_loader, device)
fasterrcnn_none_time = time.time() - start_time
print(f"Faster R-CNN_None mAP: {fasterrcnn_none_map:.3f}")
print(f"Faster R-CNN_None costs time: {fasterrcnn_none_time:.2f}s")

print("Evaluating Faster R-CNN...")
start_time = time.time()
fasterrcnn_map = evaluate(fasterrcnn_model, test_loader, device)
fasterrcnn_time = time.time() - start_time
print(f"Faster R-CNN mAP: {fasterrcnn_map:.3f}")
print(f"Faster R-CNN costs time: {fasterrcnn_time:.2f}s")

print("Evaluating Mask R-CNN...")
start_time = time.time()
maskrcnn_map = evaluate(maskrcnn_model, test_loader, device)
maskrcnn_time = time.time() - start_time
print(f"Mask R-CNN mAP: {maskrcnn_map:.3f}")
print(f"Mask R-CNN costs time: {maskrcnn_time:.2f}s")
