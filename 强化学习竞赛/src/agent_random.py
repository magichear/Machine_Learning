import numpy as np
from ConnectXState import ConnectXState


def agent_random(observation, configuration):
    """
    1. 如果存在能立即获胜的落子（自己连3且有空位），立即执行；
    2. 否则如果对手有连3威胁，在对应列阻挡；
    3. 否则在所有可行动中随机选择一列落子。
    """
    if isinstance(observation, np.ndarray):
        board = observation[:-1].tolist()
        current_player = int(observation[-1])
        columns = configuration["columns"]
        rows = configuration["rows"]
        inarow = configuration["inarow"]
    elif isinstance(observation, dict):
        board = observation["board"]
        current_player = observation.get("mark", observation.get("current_player"))
        if isinstance(configuration, dict):
            columns = configuration["columns"]
            rows = configuration["rows"]
            inarow = configuration["inarow"]
        else:
            columns, rows, inarow = (
                configuration.columns,
                configuration.rows,
                configuration.inarow,
            )
    else:
        raise ValueError(f"Unsupported observation format: {type(observation)}")

    # 游戏状态
    state = ConnectXState(board, columns, rows, inarow, current_player)

    # 检测所有威胁（已按得分降序）
    threats = state.detect_all_threats()

    # 优先查找“win”类型
    for ttype, action, _ in threats:
        if ttype == "win":
            return int(action)

    # 赢不了再查找“block”类型
    for ttype, action, _ in threats:
        if ttype == "block":
            return int(action)

    # 否则随机落子
    valid = state.get_valid_actions()
    return int(np.random.choice(valid))
