import numpy as np
import matplotlib.pyplot as plt
import cv2


def read_image(filepath="./ustc-cow.png"):
    img = cv2.imread(filepath)
    # 将图片从 BGR 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class KMeans:
    def __init__(self, k=4, max_iter=100_000_000):
        """
        这里最大迭代次数固定大一点，影响最终迭代次数的还是k值，
        小了之后对于大k值可能迭代不完
        """
        self.k = k
        self.max_iter = max_iter
        self.iter_cnt = 0

    # 随机初始化中心点
    def initialize_centers(self, points):
        """
        points: (n_samples, n_dims,)
        """
        n, d = points.shape

        centers = np.zeros((self.k, d))
        for k in range(self.k):
            # 使用更多随机点初始化中心，使kmeans更稳定
            random_index = np.random.choice(n, size=10, replace=False)
            centers[k] = points[random_index].mean(axis=0)

        return centers

    # 将每个点分配到最近的中心
    def assign_points(self, centers, points):
        """
        centers: (n_clusters, n_dims,)
        points: (n_samples, n_dims,)
        return labels: (n_samples, )
        """
        n_samples, n_dims = points.shape
        labels = np.zeros(n_samples)
        # TODO: 计算每个点与每个中心的距离并将每个点分配到最近的中心
        distances = np.sqrt(((points[:, np.newaxis] - centers) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        return labels

    # 根据点的新分配更新中心
    def update_centers(self, centers, labels, points):
        """
        centers: (n_clusters, n_dims,)
        labels: (n_samples, )
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        """
        # TODO: 根据点的新分配更新中心
        new_centers = np.zeros_like(centers)
        for i in range(self.k):
            cluster = points[labels == i]
            if len(cluster) != 0:
                new_centers[i] = cluster.mean(axis=0)  # 对每个维度求均值 ---> (n_dims,)
        return new_centers

    # k-means 聚类
    def fit(self, points):
        """
        points: (n_samples, n_dims,)
        return centers: (n_clusters, n_dims,)
        循环更新每个聚类的中心点，每次都将待分配点分配到最近的中心
        """
        centers = self.initialize_centers(points)
        for i in range(self.max_iter):
            labels = self.assign_points(centers, points)
            new_centers = self.update_centers(centers, labels, points)
            if np.allclose(centers, new_centers):
                self.iter_cnt = i + 1
                break
            centers = new_centers
        else:
            self.iter_cnt = self.max_iter
        return centers

    def compress(self, img):
        """
        img: (width, height, 3)
        return compressed img: (width, height, 3)
        k影响图片的颜色种类
        """
        # 展平图像像素
        points = img.reshape((-1, img.shape[-1]))
        # TODO: 对点进行拟合，并将每个像素值替换为其附近的中心值
        centers = self.fit(points)
        labels = self.assign_points(centers, points).astype(int)
        compressed_points = centers[labels]
        return compressed_points.reshape(img.shape)

    @staticmethod
    def compare_k(img, k_values, output_path="./comparison.png"):
        """
        img: (width, height, 3)
        k_values: [k1, k2, ...]
        output_path:
        k较小时有点卡通的感觉，大了之后逐渐向原图靠近
        """
        plt.figure(figsize=(15, 10))

        # 显示原始图像
        _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        original_size = len(buffer)  # 原始图像的存储大小（字节）
        plt.subplot(2, len(k_values) // 2 + 1, 1)
        plt.imshow(img)
        plt.title(f"Original Image\nSize: {original_size / 1024:.2f} KB")
        plt.axis("off")

        # 显示不同 k 值下的压缩图像
        for i, k in enumerate(k_values, start=2):
            kmeans = KMeans(k=k)
            compressed_img = kmeans.compress(img).round().astype(np.uint8)

            # 计算压缩图像的存储大小
            _, buffer = cv2.imencode(
                ".png", cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR)
            )
            compressed_size = len(buffer)  # 压缩图像的存储大小（字节）

            plt.subplot(2, len(k_values) // 2 + 1, i)
            plt.imshow(compressed_img)
            plt.title(
                f"k={k}\nSize: {compressed_size / 1024:.2f} KB | Iter: {kmeans.iter_cnt}"
            )
            plt.axis("off")

        # 保存对比图
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()


def test1(img, k=8):
    # 单次压缩并绘图
    kmeans = KMeans(k=k, max_iter=10)
    compressed_img = kmeans.compress(img).round().astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(compressed_img)
    plt.title("Compressed Image")
    plt.axis("off")
    plt.savefig("./compressed_image.png")


def test2(img):
    # 多种k值压缩对比
    k_values = [2, 4, 8, 16, 32, 64, 128]
    KMeans.compare_k(img, k_values)


if __name__ == "__main__":
    img = read_image(filepath="./ustc-cow.png")
    test2(img)
