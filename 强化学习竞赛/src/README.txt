train.ipynb 为各模块的实际调用，内含训练与测试逻辑

agent开头的是我为模型专门设计的对手，agent_minmax的后缀表示搜索深度

ConnectX开头的为本任务的框架，其中
	GymEnv 为 gymnasium  环境包装
	State      为棋盘状态类
	Trainer   为 stable_baseline3 库的训练器包装