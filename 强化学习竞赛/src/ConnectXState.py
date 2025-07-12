import numpy as np


class ConnectXState:
    """ConnectX游戏状态类，处理棋盘状态、动作应用和胜负判断"""

    def __init__(
        self,
        board,
        columns,
        rows,
        inarow,
        current_player=1,
        last_action=None,
        target_player=1,
    ):
        self.board = board[:]  # 一维列表表示棋盘
        self.columns = columns  # 列数
        self.rows = rows  # 行数
        self.inarow = inarow  # 连成几个棋子获胜
        self.current_player = current_player  # 当前玩家（1或2）
        self.target_player = target_player  # 用于分数计算，即最终训练出的模型在本局角色（因为每步之后，新状态中都会交换当前玩家）
        self.last_action = last_action  # 上一次动作

        self.win_reward = 1000

    def change_player(self, player):
        return 3 - player

    def get_valid_actions(self):
        """返回所有可行动作（未满的列）"""
        return [c for c in range(self.columns) if self.board[c] == 0]

    def apply_action(self, action):
        """应用动作，返回新状态（交换玩家）"""
        if action not in self.get_valid_actions():
            return None  # 无效动作

        new_board = self.board[:]
        # 从底部向上查找空位
        for r in range(self.rows - 1, -1, -1):
            if new_board[r * self.columns + action] == 0:
                new_board[r * self.columns + action] = self.current_player
                break

        return ConnectXState(
            new_board,
            self.columns,
            self.rows,
            self.inarow,
            self.change_player(self.current_player),
            action,
            target_player=self.target_player,
        )

    def is_terminal(self):
        """检查是否为终局状态"""
        return self.__check_terminal_reward()

    def __check_terminal_reward(self):
        """检查终局奖励: None为未结束"""
        # 检查每个位置是否形成连线
        for r in range(self.rows):
            for c in range(self.columns):
                if self.board[r * self.columns + c] == 0:
                    continue
                if self.__check_winner(r, c):
                    player = self.board[r * self.columns + c]
                    # 目标获胜分数为正，否则为负
                    reward = (
                        self.win_reward
                        if player == self.target_player
                        else -self.win_reward
                    )
                    return reward, player

        # 检查平局
        return (
            (self.win_reward * 0.05, 0) if len(self.get_valid_actions()) == 0 else None
        )

    def __check_winner(self, row, col):
        """检查从'指定位置'是否形成连线（即是否有胜者）"""
        player = self.board[row * self.columns + col]
        if player == 0:
            return False

        # 四个方向：水平、垂直、对角
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # 当前位置算1个
            # 向正方向检查
            for i in range(1, self.inarow):
                nr, nc = row + i * dr, col + i * dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.columns
                    and self.board[nr * self.columns + nc] == player
                ):
                    count += 1
                else:
                    break
            # 向负方向检查
            for i in range(1, self.inarow):
                nr, nc = row - i * dr, col - i * dc
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.columns
                    and self.board[nr * self.columns + nc] == player
                ):
                    count += 1
                else:
                    break

            if count >= self.inarow:
                return True
        return False

    def detect_all_threats(self, cur_action=None):
        """检测所有威胁，包括直接获胜、阻挡对手、以及避免给对手创造机会"""
        threats = []
        # 为训练作支持，因为训练时只有一个动作产生，那么就检测这个动作进行导致的后果
        if cur_action:
            valid_actions = [cur_action]
        else:
            # minmax需要评估所有下一步可能动作
            valid_actions = self.get_valid_actions()

        key_reward = self.win_reward * 0.8

        for action in valid_actions:
            # 模拟当前玩家放置棋子  放置后new_state中的current_player会变为对手，但目标不变
            new_state = self.apply_action(action)
            if not new_state:
                continue

            # 自己直接获胜
            self_win = False
            if new_state.is_terminal():
                reward_type = new_state.__check_terminal_reward()
                if reward_type:
                    threats.append(("win", action, reward_type[0]))
                    self_win = True

            # 我们的放置是否阻挡了对手的直接获胜（三缺一）
            if len(valid_actions) > 0:
                # 假设这一步是对手在操作
                self.current_player = self.change_player(self.current_player)
                suppose_state = self.apply_action(action)
                suppose_reward_type = suppose_state.__check_terminal_reward()
                if suppose_reward_type:
                    threats.append(("block", action, key_reward))

                # 恢复玩家
                self.current_player = self.change_player(self.current_player)

            # 到这里才跳过，因为让自己获胜且同时能阻挡对手获胜的行为应该得到鼓励，而自己赢之后对手就没有下一步了
            # 不放到最后是因为，模型可能会“养寇自重”，放弃直接获胜机会拿到更多过程分
            if self_win:
                continue

            # 检查是否给对手创造了直接获胜机会（送台阶、不格挡，应该绝对避免）
            danger_score = 0
            for opp_action in new_state.get_valid_actions():
                opp_state = new_state.apply_action(opp_action)
                if opp_state:
                    opp_reward_type = opp_state.__check_terminal_reward()
                    if opp_reward_type and opp_reward_type[1] != self.target_player:
                        danger_score -= key_reward

            if danger_score < 0:
                threats.append(("avoid", action, danger_score))
        # 按分数降序
        threats.sort(key=lambda x: x[2], reverse=True)
        return threats

    def model_apply(self, action=None):
        """
        计算奖励并应用动作（专为模型封装），返回奖励具体数值（密集奖励，废弃）
        """

        # 先评估环境，接着评估动作可能导致的结果，最后应用动作
        position_score = self.evaluate_position()
        threats = self.detect_all_threats(cur_action=action)
        self.apply_action(action)

        return {
            "reward": position_score,
            "is_terminal": False,
            "winner": -1,
            "threats": threats,
            "position_score": position_score,
        }

    def evaluate_position(self):
        """评估当前局面，返回评估分数"""
        score = 0

        # 为每个玩家计算威胁分数
        for player in [1, 2]:
            player_score = 0

            # 检查所有可能的连线位置
            for r in range(self.rows):
                for c in range(self.columns):
                    # 检查四个方向的连线潜力
                    directions = [
                        (0, 1),
                        (1, 0),
                        (1, 1),
                        (1, -1),
                    ]  # 水平、垂直、两个对角

                    for dr, dc in directions:
                        window_score = self.__evaluate_window(r, c, dr, dc, player)
                        player_score += window_score

            # 目标玩家的分数为正，对手的分数为负
            if player == self.target_player:
                score += player_score
            else:
                score -= player_score

        # 引入随机噪声
        return score + np.random.uniform(-0.1, 0.1)

    def __evaluate_window(self, row, col, dr, dc, player):
        """评估从指定位置开始的窗口（长度为inarow），考虑重力约束"""
        window = []
        positions = []

        # 收集窗口内的棋子和位置
        for i in range(self.inarow):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                window.append(self.board[r * self.columns + c])
                positions.append((r, c))
            else:
                return 0  # 窗口超出边界

        return self.__score_window_with_gravity(window, positions, player)

    def __score_window_with_gravity(self, window, positions, player):
        """为窗口打分，考虑重力约束"""
        score = 0
        opponent = self.change_player(player)

        player_count = window.count(player)
        opponent_count = window.count(opponent)

        # 如果对手已经占据了窗口，这个窗口对当前玩家无价值
        if opponent_count > 0:
            return 0

        # 检查空位是否可达（考虑重力）
        reachable_empty = 0
        for i, cell in enumerate(window):
            if cell == 0:
                r, c = positions[i]
                if self.__can_reach_position(r, c):
                    reachable_empty += 1

        # 如果没有可达的空位，这个窗口无价值
        if reachable_empty == 0 and player_count < self.inarow:
            return 0

        # 根据己方棋子数量和可达空位数量打分
        if player_count == 4:
            score += self.win_reward  # 获胜（在训练时不会到达这里）
        elif player_count == 3 and reachable_empty >= 1:
            score += self.win_reward / 10  # 三连，有可达空位
        elif player_count == 2 and reachable_empty >= 2:
            score += self.win_reward / 100  # 二连，有足够可达空位
        elif player_count == 1 and reachable_empty >= 3:
            score += self.win_reward / 1000  # 一连，有足够可达空位

        return score

    def __can_reach_position(self, row, col):
        """检查指定位置是否可以通过合法落子到达"""
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return False

        # 如果该位置已有棋子，则无法到达
        if self.board[row * self.columns + col] != 0:
            return False

        # 检查该位置下方是否都被填满（重力约束）
        for r in range(row + 1, self.rows):
            if self.board[r * self.columns + col] == 0:
                return False

        return True
