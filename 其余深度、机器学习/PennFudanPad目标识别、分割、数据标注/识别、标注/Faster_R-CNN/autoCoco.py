import os
import json
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from engine import get_detection_model


def create_coco_json(image_dir, predictions_list, output_path, class_names):
    coco_data = {"images": [], "annotations": [], "categories": []}

    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append(
            {"id": idx, "name": class_name, "supercategory": "none"}
        )

    annotation_id = 1
    for image_id, (image_name, predictions) in enumerate(predictions_list):
        # 获取图片信息
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        width, height = image.size

        # 添加图片信息
        coco_data["images"].append(
            {"id": image_id, "file_name": image_name, "width": width, "height": height}
        )

        # 添加标注信息
        for box, label, score in zip(
            predictions["boxes"], predictions["labels"], predictions["scores"]
        ):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            # COCO 格式的边界框为 [x, y, width, height]
            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [float(xmin), float(ymin), float(width), float(height)],
                    "area": float(width * height),
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
            annotation_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Saved COCO annotations to {output_path}")


def main():
    # 模型和数据路径
    model_path = "../models/9.pth"
    image_dir = "../PennFudanPed/PNGImages"
    output_path = "../PennFudanPed/annotations.json"
    class_names = ["background", "person"]

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TORCH_HOME"] = "./models"
    model = get_detection_model(
        num_classes=len(class_names), is_rcnn_pretrained=False, is_mask_rcnn=False
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions_list = []

    # 遍历图片并生成预测结果
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        # 过滤低置信度的预测结果
        keep = predictions["scores"] > 0.5
        predictions = {
            "boxes": predictions["boxes"][keep].tolist(),
            "labels": predictions["labels"][keep].tolist(),
            "scores": predictions["scores"][keep].tolist(),
        }

        predictions_list.append((image_name, predictions))

    create_coco_json(image_dir, predictions_list, output_path, class_names)


if __name__ == "__main__":
    main()
