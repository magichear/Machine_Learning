import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make
import numpy as np
from ConnectXState import ConnectXState
import random


class ConnectXGymEnv(gym.Env):
    """ConnectX的Gymnasium环境包装器，兼容Stable-Baselines3"""

    def __init__(
        self, opponent_agents=["random"], columns=7, rows=6, inarow=4, target_player=1
    ):
        super().__init__()

        # 环境参数
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.opponent_agents = opponent_agents
        self.opponent_agent = random.choice(opponent_agents)
        self.invalid_punishment = -2000  # 非法动作惩罚
        self.target_player = target_player  # 目标玩家（用于奖励计算）

        # Gymnasium环境必需属性
        self.action_space = spaces.Discrete(columns)

        # 观察空间：棋盘状态 + 当前玩家标识
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(rows * columns + 1,), dtype=np.int32
        )

        # 初始化Kaggle环境用于对手
        self.kaggle_env = make("connectx", debug=False)
        self.reset()

    def reset(self, seed=None, options=None):
        """
        重置环境 - Gymnasium格式
        此处控制先后手
        """
        # 处理seed参数（Gymnasium要求）
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.kaggle_env.reset(2)  # 两个玩家

        self.current_state = ConnectXState(
            board=[0] * (self.rows * self.columns),
            columns=self.columns,
            rows=self.rows,
            inarow=self.inarow,
            current_player=1,  # 初始玩家为1（无论模型先后手）
            target_player=self.target_player,
        )
        self.done = False

        if len(self.opponent_agents) == 1:
            self.opponent_agent = self.opponent_agents[0]
        else:
            # 这里暂时只考虑最多两个对手
            self.opponent_agent = random.choices(
                self.opponent_agents,
                weights=[0.9, 0.1] + [0] * (len(self.opponent_agents) - 2),
            )

        # 后手，让对手先走一步
        if self.current_state.current_player == 2:
            opponent_action = self.__get_opponent_action()
            if opponent_action is not None:
                self.current_state = self.current_state.apply_action(opponent_action)

        # 切换先后手（在下一次reset后生效）
        self.target_player = self.current_state.change_player(self.target_player)

        # Gymnasium要求返回(observation, info)
        return self.__get_obs(), {}

    def step(self, action):
        """执行一回合"""
        if self.done:
            raise ValueError("Episode已结束，需要reset")

        # 检查动作有效性
        if action not in self.current_state.get_valid_actions():
            return (
                self.__get_obs(),
                self.invalid_punishment,
                True,
                False,  # truncated 为true表示因超时而强行终止
                {},
            )

        # 训练对象的回合
        reward = self.__calc_and_apply(action)

        # 如果没结束就让对手继续跑
        if self.current_state.is_terminal():
            self.done = True
        else:
            # 对手回合，在现有状态上选择一个动作
            opponent_action = self.__get_opponent_action()
            if opponent_action is not None:
                self.current_state = self.current_state.apply_action(opponent_action)
            if self.current_state.is_terminal():
                self.done = True

        # Gymnasium格式: (obs, reward, terminated, truncated, info)
        return self.__get_obs(), reward, self.done, False, {}

    def __get_obs(self):
        """获取当前观察状态"""
        # 棋盘状态 + 训练目标标识（转化为列表，区分先后手）
        obs = np.array(
            self.current_state.board + [self.current_state.target_player],
            dtype=np.int32,
        )
        return obs

    def __calc_and_apply(self, action):
        """
        计算奖励并应用动作
        """
        # 纯数值
        reward = self.current_state.evaluate_position()
        # 元组列表
        threats = self.current_state.detect_all_threats(cur_action=action)
        self.current_state = self.current_state.apply_action(action)

        for threat in threats:
            reward += threat[2]  # 累加威胁分数

        return reward

    def __get_opponent_action(self):
        """获取对手动作"""
        if self.opponent_agent == "random":
            possible_actions = self.current_state.get_valid_actions()
            return random.choice(possible_actions) if possible_actions else None
        elif callable(self.opponent_agent):
            # 构造observation和configuration
            observation = {
                "board": self.current_state.board,
                "mark": self.current_state.current_player,
            }
            configuration = {
                "columns": self.columns,
                "rows": self.rows,
                "inarow": self.inarow,
            }
            return self.opponent_agent(observation, configuration)
        return None
