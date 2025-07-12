# 导入必要的库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # 神经网络相关模块
import numpy as np  # 数值计算库
from torch.utils.data import DataLoader  # 处理数据加载
import torchvision
from torchvision import datasets, transforms  # 处理图像数据集和数据变换
from torchvision.utils import save_image  # 保存生成的图像
import os  # 处理文件和目录操作
from torch.utils.tensorboard import SummaryWriter  # TensorBoard


# =============================== 生成器（Generator） ===============================
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # TODO: 使用线性层将随机噪声映射到第一个隐藏层
            nn.Linear(input_dim, hidden_dim),
            # TODO: 使用 ReLU 作为激活函数，帮助模型学习非线性特征
            nn.ReLU(),
            # TODO: 使用线性层将第一个隐藏层映射到第二个隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            # TODO: 再次使用 ReLU 激活函数
            nn.ReLU(),
            # TODO: 使用线性层将第二个隐藏层映射到输出层，输出为图像的像素大小
            nn.Linear(hidden_dim, output_dim),
            # TODO:使用 Tanh 将输出归一化到 [-1, 1]，适用于图像生成
            nn.Tanh(),
        )

    def forward(self, x):
        # TODO: 前向传播: 将输入 x 通过模型进行计算，得到生成的图像
        return self.model(x)


# =============================== 判别器（Discriminator） ===============================
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # TODO: 输入层到第一个隐藏层，使用线性层
            nn.Linear(input_dim, hidden_dim),
            # TODO:  使用 LeakyReLU 激活函数，避免梯度消失问题，negative_slope参数设置为0.1
            nn.LeakyReLU(0.1),
            # TODO:  第一个隐藏层到第二个隐藏层，使用线性层
            nn.Linear(hidden_dim, hidden_dim),
            # TODO:   再次使用 LeakyReLU 激活函数，negative_slope参数设置为0.1
            nn.LeakyReLU(0.1),
            # TODO:  第二个隐藏层到输出层，使用线性层
            nn.Linear(hidden_dim, 1),
            # TODO:  使用 Sigmoid 激活函数，将输出范围限制在 [0, 1]
            nn.Sigmoid(),
        )

    def forward(self, x):
        # TODO:  前向传播: 将输入 x 通过模型进行计算，得到判别结果
        return self.model(x)


# =============================== 主函数 ===============================
def main():

    # 设备配置：使用 GPU（如果可用），否则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置模型和训练的超参数
    input_dim = 100  # 生成器输入的随机噪声向量维度
    hidden_dim = 256  # 隐藏层维度
    output_dim = 28 * 28  # MNIST 数据集图像尺寸（28x28）
    batch_size = 128  # 训练时的批量大小
    num_epoch = 200  # TODO: 训练的总轮数, 可以根据需要调整

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root="./data/", train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # TODO: 创建生成器G和判别器D，并移动到 device
    G = Generator(input_dim, hidden_dim, output_dim).to(device)
    D = Discriminator(output_dim, hidden_dim).to(device)

    # TODO: 定义针对生成器G的优化器optim_G和针对判别器D的优化器optim_D，要求使用Adam优化器，学习率设置为0.0002
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    loss_func = nn.BCELoss()  # 使用二元交叉熵损失

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir="./logs/experiment_gan")

    # 开始训练
    for epoch in range(num_epoch):
        total_loss_D, total_loss_G = 0, 0
        for i, (real_images, _) in enumerate(train_loader):
            loss_D = train_discriminator(
                real_images, D, G, loss_func, optim_D, batch_size, input_dim, device
            )  # 训练判别器
            loss_G = train_generator(
                D, G, loss_func, optim_G, batch_size, input_dim, device
            )  # 训练生成器

            total_loss_D += loss_D
            total_loss_G += loss_G

            # 每 100 步打印一次损失
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(
                    f"Epoch {epoch:02d} | Step {i + 1:04d} / {len(train_loader)} | Loss_D {total_loss_D / (i + 1):.4f} | Loss_G {total_loss_G / (i + 1):.4f}"
                )

            # 记录每个epoch的平均损失到 TensorBoard
        writer.add_scalar(
            "GAN/Loss/Discriminator", total_loss_D / len(train_loader), epoch
        )
        writer.add_scalar("GAN/Loss/Generator", total_loss_G / len(train_loader), epoch)

        # 生成并保存示例图像
        with torch.no_grad():
            noise = torch.randn(64, input_dim, device=device)
            fake_images = G(noise).view(-1, 1, 28, 28)  # 调整形状为图像格式

            # 记录生成的图像到 TensorBoard
            img_grid = torchvision.utils.make_grid(fake_images, normalize=True)
            writer.add_image("Generated Images", img_grid, epoch)


# =============================== 训练判别器 ===============================
def train_discriminator(
    real_images, D, G, loss_func, optim_D, batch_size, input_dim, device
):
    """训练判别器"""
    real_images = real_images.view(-1, 28 * 28).to(device)  # 获取真实图像并展平
    real_output = D(real_images)  # 判别器预测真实图像
    # TODO:   # 计算真实样本的损失real_loss
    real_labels = torch.ones(real_output.size(), device=device)
    real_loss = loss_func(real_output, real_labels)

    noise = torch.randn(batch_size, input_dim, device=device)  # 生成随机噪声
    fake_images = G(noise).detach()  # 生成假图像（detach 避免梯度传递给 G）
    fake_output = D(fake_images)  # 判别器预测假图像
    # TODO:   # 计算假样本的损失fake_loss
    fake_labels = torch.zeros(fake_output.size(), device=device)
    fake_loss = loss_func(fake_output, fake_labels)

    loss_D = real_loss + fake_loss  # 判别器总损失
    optim_D.zero_grad()  # 清空梯度
    loss_D.backward()  # 反向传播
    optim_D.step()  # 更新判别器参数

    return loss_D.item()  # 返回标量损失


# =============================== 训练生成器 ===============================
def train_generator(D, G, loss_func, optim_G, batch_size, input_dim, device):
    """训练生成器"""
    noise = torch.randn(batch_size, input_dim, device=device)  # 生成随机噪声
    fake_images = G(noise)  # 生成假图像
    fake_output = D(fake_images)  # 判别器对假图像的判断
    # TODO: 计算生成器损失（希望生成的图像判别为真）
    real_labels = torch.ones(fake_output.size(), device=device)
    loss_G = loss_func(fake_output, real_labels)

    optim_G.zero_grad()  # 清空梯度
    loss_G.backward()  # 反向传播
    optim_G.step()  # 更新生成器参数

    return loss_G.item()  # 返回标量损失


if __name__ == "__main__":
    main()

# 程序执行完后，使用 TensorBoard 可视化训练过程：
# 在终端使用命令 tensorboard --logdir=./logs/experiment_gan  --samples_per_plugin=images=1000
