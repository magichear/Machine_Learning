import os
import torch
import numpy as np

import time

from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.metrics import auc

from utils.evaluation import evaluate_sample
from PIL import ImageDraw, Image
from torchvision.transforms import ToPILImage


def get_detection_model(num_classes=2, is_rcnn_pretrained=False, is_mask_rcnn=False):
    if is_mask_rcnn:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if is_rcnn_pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights)
    else:
        weights = (
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT if is_rcnn_pretrained else None
        )
        model = fasterrcnn_resnet50_fpn(weights=weights)

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    count = 0
    global_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device).float() for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        dict_loss = model(images, targets)
        losses = sum(loss for loss in dict_loss.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        count += 1
        global_loss += float(losses.cpu().detach().numpy())

        if count % 10 == 0:
            print(
                "Loss value after {} batches is {}".format(
                    count, round(global_loss / count, 2)
                )
            )

    return global_loss


def draw_boxes(image, boxes, labels):
    if isinstance(image, torch.Tensor):
        image = ToPILImage()(image)

    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), f"Label: {label}", fill="red")
    return image


def draw_masks(image, masks):
    if isinstance(image, torch.Tensor):
        image = ToPILImage()(image)

    for mask in masks:
        mask = (mask[0] > 0.5).astype(np.uint8)  # 二值化掩码
        mask_image = Image.fromarray(mask * 255).convert("L")
        image.paste(mask_image, (0, 0), mask_image)
    return image


def train(
    model,
    num_epochs,
    train_loader,
    test_loader,
    optimizer,
    device,
    save_path,
    dataset_test,
    is_mask_rcnn=False,
):
    """
    Train the model
    :param model:
    :param num_epochs:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param device:
    :param save_path
    :return:
    """
    for epoch in range(num_epochs):
        print("epoch {}/{}..".format(epoch, num_epochs))
        start = time.time()
        train_one_epoch(model, optimizer, train_loader, device)
        mAP = evaluate(model, test_loader, device=device, epoch_num=epoch)
        end = time.time()

        print("epoch {} done in {}s".format(epoch, round(end - start, 2)))
        print("mAP after epoch {} is {}:".format(epoch, round(mAP, 3)))

        if (epoch + 1) % 5 == 0:
            print(
                "Model saved to {}".format(os.path.join(save_path, str(epoch) + ".pth"))
            )
            torch.save(model.state_dict(), os.path.join(save_path, str(epoch) + ".pth"))

        print("#" * 25)

        # 每个 epoch 后，使用测试集的一张图片进行预测
        model.eval()
        with torch.no_grad():
            img, _ = dataset_test[10]
            img_tensor = img.to(device).unsqueeze(0)
            prediction = model(img_tensor)[0]

            # 获取预测的边界框和标签
            boxes = prediction["boxes"].cpu().numpy()
            labels = prediction["labels"].cpu().numpy()
            # 绘制边界框
            img_with_boxes = draw_boxes(img, boxes, labels)
            if is_mask_rcnn:
                masks = prediction["masks"].cpu().numpy()
                # 绘制掩码
                img_with_mask = draw_masks(img_with_boxes, masks)
                img_with_mask.save(f"epoch_{epoch + 1}_mask.png")

            img_with_boxes.save(f"epoch_{epoch + 1}_prediction.png")
            print(f"Saved prediction for epoch {epoch + 1}.")

    torch.save(model.state_dict(), os.path.join(save_path, str(num_epochs) + ".pth"))


def evaluate(model, test_loader, device, epoch_num=0):
    """
    Computes precision and recall for a given trehsold (default = 0.5)
    :param model :
    :param test_loader:
    :param device:
    :return : tuple containing precision and recall
    """
    results = []
    model.eval()
    nbr_boxes = 0

    with torch.no_grad():
        for batch, (images, targets_true) in enumerate(test_loader):
            images = list(image.to(device).float() for image in images)
            targets_pred = model(images)

            targets_true = [
                {k: v.cpu().float() for k, v in t.items()} for t in targets_true
            ]
            targets_pred = [
                {k: v.cpu().float() for k, v in t.items()} for t in targets_pred
            ]

            os.makedirs("results", exist_ok=True)

            for i in range(len(targets_true)):
                target_true = targets_true[i]
                target_pred = targets_pred[i]
                nbr_boxes += target_true["labels"].shape[0]

                evaluate_result = evaluate_sample(target_pred, target_true)
                results = results + evaluate_result

    results = sorted(results, key=lambda k: k["score"], reverse=True)

    acc_TP = np.zeros(len(results))
    acc_FP = np.zeros(len(results))
    recall = np.zeros(len(results))
    precision = np.zeros(len(results))

    if results[0]["TP"] == 1:
        acc_TP[0] = 1
    else:
        acc_FP[0] = 1

    for ii in range(1, len(results)):
        acc_TP[ii] = results[ii]["TP"] + acc_TP[ii - 1]
        acc_FP[ii] = (1 - results[ii]["TP"]) + acc_FP[ii - 1]

        precision[ii] = acc_TP[ii] / (acc_TP[ii] + acc_FP[ii])
        recall[ii] = acc_TP[ii] / nbr_boxes

    return auc(recall, precision)
