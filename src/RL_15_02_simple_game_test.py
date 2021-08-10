# 載入相關套件
from RL_15_01_simple_game import Environment, Agent

# 建立實驗，含環境、代理人物件
env = Environment()
agent = Agent()

# 進行實驗
for _ in range(10):
    env.__init__()  # 重置
    total_reward=0  # 累計報酬
    while not env.is_done():
        # 採取行動
        action = agent.action(env)
        
        # 更新下一步
        state, reward = env.step(action)
        
        # 計算累計報酬
        total_reward += reward

    # 顯示累計報酬
    print(f"累計報酬: {total_reward:.4f}")
