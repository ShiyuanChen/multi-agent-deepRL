from envs.multiAgentGrid import multiAgentEnv
import numpy as np
import matplotlib.pyplot as plt

env = multiAgentEnv(num_agents=5, sensor_range=(10,10))

env.printInfo()
x = env.resetAll()
print(env.agent_pos)
print(x.shape)
env.render()

plt.imshow(x[0])
plt.show()
action = [0, 0, 0, 0, 0]
next_state, unclipped_reward, done, info = env.step(action)
env.render()
