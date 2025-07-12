import os
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- 配置参数 ---
ORL_DATASET_PATH = "./orl_faces"
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 92
TEST_SIZE_RATIO = 0.3
N_COMPONENTS = [
    10,
    20,
    40,
    70,
    100,
    200,
    (int)(
        min(400 * (1 - TEST_SIZE_RATIO), IMAGE_HEIGHT * IMAGE_WIDTH)
    ),  # PCA的主成分数量由协方差矩阵的秩（受限于行数与列数）决定
]
N_COMPONENTS_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RANDOM_STATE_SEED = 42


def load_orl_dataset(dataset_path):
    images_list = []
    labels_list = []
    subject_folders = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if d.startswith("s") and os.path.isdir(os.path.join(dataset_path, d))
        ]
    )

    # 假设h, w能从第一张成功加载的图片中获取
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
    first_image_loaded = False

    for subject_id, subject_folder in enumerate(subject_folders):
        folder_path = os.path.join(dataset_path, subject_folder)
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pgm")])
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path) as img:
                img_gray = img.convert("L")
                if not first_image_loaded:  # 从第一张成功加载的图片获取实际尺寸
                    w_actual, h_actual = img_gray.size
                    if h_actual != h or w_actual != w:  # 如果预设和实际不同，则更新
                        print(
                            f"注意: 图像实际尺寸 ({h_actual}x{w_actual}) 与预设 ({h}x{w}) 不同，将使用实际尺寸。"
                        )
                        h, w = h_actual, w_actual
                    first_image_loaded = True

                # 将图像转换为numpy数组 (np.float32)并扁平化为1D向量。
                img_array_flat = np.array(img, dtype=np.float32).flatten()  # 扁平化
                images_list.append(img_array_flat)
                labels_list.append(subject_id)

    print(f"数据集加载完毕: 共加载 {len(images_list)} 张图像。")
    return np.array(images_list), np.array(labels_list), h, w


def plot_gallery(title, images, image_h, image_w, n_row=3, n_col=5, cmap=plt.cm.gray):
    # (plot_gallery 函数保持不变，它对于可视化结果仍然有用)
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        if i >= n_row * n_col:
            break
        plt.subplot(n_row, n_col, i + 1)
        img_to_show = (
            comp.reshape((image_h, image_w))
            if comp.ndim == 1 and comp.shape[0] == image_h * image_w
            else comp
        )
        vmax = (
            max(img_to_show.max(), -img_to_show.min())
            if img_to_show.ndim == 2
            else None
        )
        plt.imshow(
            img_to_show,
            cmap=cmap,
            interpolation="nearest",
            vmin=(-vmax if vmax is not None else None),
            vmax=vmax,
        )
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def main(n_components=10):
    # 1. 加载数据集 (移除了外部的路径检查和try-except)
    X, y, h, w = load_orl_dataset(ORL_DATASET_PATH)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE_RATIO, stratify=y, random_state=RANDOM_STATE_SEED
    )

    # 3. PCA计算Eigenfaces
    # TODO: 实例化 PCA 对象，并使用训练集数据 X_train 训练 PCA 模型。这里使用了随机 SVD 方法，并且不进行白化处理。
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False)
    pca.fit(X_train)

    # TODO: 可视化平均脸和一些Eigenfaces
    eigenfaces = pca.components_.reshape(-1, h, w)
    mean_face = pca.mean_.reshape((h, w))

    plot_gallery("Mean Face", [mean_face], h, w, n_row=1, n_col=1)
    plot_gallery(
        f"Top Eigenfaces", eigenfaces[: min(n_components, 10)], h, w, n_row=2, n_col=5
    )

    # 4. 数据投影到Eigenface空间
    # TODO:  使用训练好的 PCA 模型将训练集 X_train 和测试集 X_test 投影到低维空间。
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # /结束 TODO (4)

    # 5. 训练KNN分类器
    # TODO: 实例化 KNeighborsClassifier 对象，并使用降维后的训练数据和对应的标签训练 KNN 分类器。使用 distance 权重和 euclidean 距离。
    knn_classifier = KNeighborsClassifier(
        n_neighbors=1, weights="distance", metric="euclidean"
    )
    knn_classifier.fit(X_train_pca, y_train)

    # 6. 进行预测并评估模型
    # TODO: 使用训练好的 KNN 分类器对降维后的测试数据进行预测。
    y_pred = knn_classifier.predict(X_test_pca)

    # 计算准确率
    # TODO: 计算预测结果与真实标签之间的准确率
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n模型准确率: {accuracy * 100:.2f}%")
    print("分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7. 可视化部分测试结果
    X_test_reconstructed = pca.inverse_transform(X_test_pca)
    num_to_show = min(5, len(X_test))
    if num_to_show > 0:
        sample_indices = np.random.choice(len(X_test), size=num_to_show, replace=False)
        orig_vs_recon_images = []
        orig_vs_recon_titles = []
        for i in sample_indices:
            true_l, pred_l = y_test[i], y_pred[i]
            orig_vs_recon_images.extend(
                [X_test[i].reshape((h, w)), X_test_reconstructed[i].reshape((h, w))]
            )
            orig_vs_recon_titles.extend(
                [
                    f"Original: P{true_l+1}",
                    f"Recon.\nPred: P{pred_l+1} ({'OK' if true_l==pred_l else 'NG'})",
                ]
            )
        plot_gallery(
            f"Test Samples - Acc: {accuracy*100:.2f}%",
            orig_vs_recon_images,
            h,
            w,
            n_row=num_to_show,
            n_col=2,
        )
    return accuracy


def test_all():  # 运行前最好注释可视化
    accuracies_n_components = []
    accuracies_n_components_2 = []

    # 测试 N_COMPONENTS_2
    for n_component in N_COMPONENTS_2:
        print(f"\n正在测试 n_components={n_component}...")
        accuracy = main(n_component)
        accuracies_n_components_2.append(accuracy)

    # 测试 N_COMPONENTS
    for n_component in N_COMPONENTS:
        print(f"\n正在测试 n_components={n_component}...")
        accuracy = main(n_component)
        accuracies_n_components.append(accuracy)

    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左子图：N_COMPONENTS_2 的准确率折线图
    axes[0].plot(
        N_COMPONENTS_2, accuracies_n_components_2, marker="o", label="Accuracy"
    )
    for i, acc in enumerate(accuracies_n_components_2):
        axes[0].text(
            N_COMPONENTS_2[i], acc, f"{acc:.4f}", fontsize=8, ha="center", va="bottom"
        )
    axes[0].set_title("Accuracy vs. N_COMPONENTS_2")
    axes[0].set_xlabel("Number of Components (N_COMPONENTS_2)")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    # 右子图：N_COMPONENTS 的准确率折线图
    axes[1].plot(N_COMPONENTS, accuracies_n_components, marker="o", label="Accuracy")
    for i, acc in enumerate(accuracies_n_components):
        axes[1].text(
            N_COMPONENTS[i], acc, f"{acc:.4f}", fontsize=8, ha="center", va="bottom"
        )
    axes[1].set_title("Accuracy vs. N_COMPONENTS")
    axes[1].set_xlabel("Number of Components (N_COMPONENTS)")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("N_COMPONENTS_Compare.png", dpi=300)
    plt.show()


def test1(n_components=10):
    main(n_components)


if __name__ == "__main__":
    test1(50)
