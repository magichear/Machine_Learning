import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from ConnectXGymEnv import ConnectXGymEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.vec_env import DummyVecEnv


class ConnectXTrainer:
    def __init__(self, model_type="PPO", opponents=["random"]):
        self.model_type = model_type
        self.opponents = opponents
        self.model = None
        self.model_path = "./connectx_models/"
        self.monitor_log_path = "./connectx_logs/"
        self.env = self.create_environment()

    def create_environment(self):
        env = ConnectXGymEnv(opponent_agents=self.opponents)
        env = Monitor(env, self.monitor_log_path, allow_early_resets=True)
        self.env = DummyVecEnv([lambda: env])
        return self.env

    def create_model(self, tensorboard_log="./connectx_tensorboard/"):
        if self.model_type == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                tensorboard_log=tensorboard_log,
            )
        elif self.model_type == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                batch_size=128,
                tensorboard_log=tensorboard_log,
            )
        return self.model

    def train(self, total_timesteps=50000, tensorboard_log="./connectx_tensorboard/"):
        if self.model is None:
            self.create_model(tensorboard_log)

        # 设置评估回调，保存最佳模型
        eval_env = Monitor(
            ConnectXGymEnv(opponent_agents=self.opponents),
            self.monitor_log_path + "eval/",
            allow_early_resets=True,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./connectx_models/",
            eval_freq=int(total_timesteps / 5),
            log_path="./connectx_logs/",
            deterministic=True,
            render=False,
        )

        # 每 int(total_timesteps / 5) 轮保存一次
        save_freq = int(total_timesteps / 5)
        for step in range(0, total_timesteps, save_freq):
            self.model.learn(
                total_timesteps=save_freq,
                callback=eval_callback,
                reset_num_timesteps=False,
            )
            model_save_name = f"{self.model_path}{self.model_type.lower()}_{save_freq}_{step+save_freq}.zip"
            self.model.save(model_save_name)
            print(f"模型已保存为: {model_save_name}")

        # 保存最终模型
        self.model.save(f"{self.model_path}{self.model_type.lower()}_final")
        print(f"模型已保存为: {self.model_type.lower()}_final")
        all_models = os.listdir(self.model_path)
        all_models = [os.path.join(self.model_path, model) for model in all_models]
        return self.model, all_models

    def show(self, window_size=1000):
        try:
            df = load_results(self.monitor_log_path)
            actual_window = min(window_size, len(df))

            # 绘制奖励曲线
            plt.figure(figsize=(12, 4))
            # 原始奖励
            plt.subplot(1, 2, 1)
            plt.plot(df["r"], alpha=0.3, label="Raw Rewards")
            if len(df) >= actual_window:
                plt.plot(
                    df["r"].rolling(window=actual_window).mean(),
                    label=f"Moving Average ({actual_window})",
                )
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Rewards")
            plt.legend()
            plt.grid(True)

            # 累积奖励
            plt.subplot(1, 2, 2)
            cumulative_rewards = df["r"].cumsum()
            plt.plot(cumulative_rewards, label="Cumulative Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title("Cumulative Rewards")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # 统计信息
            print(f"=================================================")
            print(f"总回合数: {len(df)}")
            print(f"平均奖励: {df['r'].mean():.2f}")
            print(f"最高奖励: {df['r'].max():.2f}")
            print(f"最低奖励: {df['r'].min():.2f}")
            print(
                f"最后{min(100, len(df))}回合平均奖励: {df['r'].tail(min(100, len(df))).mean():.2f}"
            )

        except Exception as e:
            print(e)
