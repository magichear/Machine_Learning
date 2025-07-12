from ConnectXState import ConnectXState
import numpy as np


# 缓存计算过的状态，优化迭代加深效率
__cache = {}


def __potential_actions(state):
    """优先考虑中心列（四通八达）"""
    actions = state.get_valid_actions()
    center_column = state.columns // 2
    actions.sort(key=lambda a: abs(a - center_column))
    return actions


def __minmax(state, depth, alpha, beta, maximizing_player):
    # 检查缓存
    state_key = (tuple(state.board), state.current_player, depth, maximizing_player)
    if state_key in __cache:
        return __cache[state_key]

    if depth == 0 or state.is_terminal():
        eval_score = state.evaluate_position()
        __cache[state_key] = (eval_score, None)
        return eval_score, None

    # 检测所有威胁（排降序，取得分最大的）
    threats = state.detect_all_threats()

    # 如果有获胜机会，立即采取
    win_actions = [action for threat_type, action, _ in threats if threat_type == "win"]
    if win_actions:
        best_action = win_actions[0]
        eval_score = 1000 if maximizing_player else -1000
        __cache[state_key] = (eval_score, best_action)
        return eval_score, best_action

    # 如果对手有直接获胜威胁，必须阻挡
    block_actions = [
        action for threat_type, action, _ in threats if threat_type == "block"
    ]

    # 避免危险动作（跟上面有些重复）
    avoid_actions = set(
        [action for threat_type, action, _ in threats if threat_type == "avoid"]
    )

    # 获取所有可能动作，优先考虑格挡，避免危险动作
    all_actions = __potential_actions(state)
    if block_actions:
        priority_actions = block_actions
    else:
        priority_actions = [
            action for action in all_actions if action not in avoid_actions
        ]
        # 如果所有动作都危险，开摆就行了，游戏结束（不过还是可以赌一下，选威胁最小的）
        if not priority_actions:
            priority_actions = avoid_actions

    if maximizing_player:
        max_eval = float("-inf")
        best_action = None
        for action in priority_actions:
            child_state = state.apply_action(action)
            if child_state:
                eval_score, _ = __minmax(child_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        __cache[state_key] = (max_eval, best_action)
        return max_eval, best_action
    else:
        min_eval = float("inf")
        best_action = None
        for action in priority_actions:
            child_state = state.apply_action(action)
            if child_state:
                eval_score, _ = __minmax(child_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
        __cache[state_key] = (min_eval, best_action)
        return min_eval, best_action


def agent_minmax(observation, configuration):
    # 训练环境（NumPy 数组）
    if isinstance(observation, np.ndarray):
        board = observation[:-1]  # 棋盘状态（去掉当前玩家标记）
        current_player = observation[-1]
        columns = configuration["columns"]
        rows = configuration["rows"]
        inarow = configuration["inarow"]
    elif isinstance(observation, dict):
        # 评估环境（字典）
        board = observation["board"]
        current_player = observation["mark"]
        columns = (
            configuration["columns"]
            if isinstance(configuration, dict)
            else configuration.columns
        )
        rows = (
            configuration["rows"]
            if isinstance(configuration, dict)
            else configuration.rows
        )
        inarow = (
            configuration["inarow"]
            if isinstance(configuration, dict)
            else configuration.inarow
        )
    else:
        raise ValueError("Unsupported observation format")

    # 当前状态
    state = ConnectXState(board, columns, rows, inarow, current_player)

    # 如果棋局为空或只有一子，随机决策
    if sum(board) <= 1:
        return int(np.random.choice(state.get_valid_actions()))

    # 根据剩余可用步数动态调整深度
    # remaining_moves = len(state.get_valid_actions())
    max_depth = 1

    # 迭代加深
    best_action = None
    for depth in range(1, max_depth + 1):
        _, action = __minmax(
            state, depth, float("-inf"), float("inf"), current_player == 1
        )
        if action is not None:
            best_action = action

    return best_action


def agent(observation, configuration):
    return agent_minmax(observation, configuration)
