import numpy as np
import matplotlib.pyplot as plt

class actionSpace(object):
    def __init__(self, num_actions=4):
        self.n = num_actions


class multiAgentEnv(object):
    def __init__(self, grid_size=(100,100), num_actions=4, num_agents=5, obstacle_numrange=(5,10), obstacle_sizerange=(2, 20), sensor_range=(10,10)):    
        assert(num_actions > 0 and num_agents > 0 and obstacle_numrange[0] <= obstacle_numrange[1] and
               obstacle_sizerange[0] <= obstacle_sizerange[1] and sensor_range[0] >= 0 and sensor_range[1] >= 0)

        self.grid_size = grid_size
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.obstacle_numrange = obstacle_numrange
        self.obstacle_sizerange = obstacle_sizerange
        self.sensor_range = sensor_range
        self.action_space = actionSpace(num_actions)
        self.directions = [1, 0, -1, 0, 1]
        self.resetted = False

    def printInfo(self):
        print('grid_size: ', self.grid_size)
        print('num_actions: ', self.num_actions)
        print('num_agents: ', self.num_agents)
        print('obstacle_sizerange: ', self.obstacle_sizerange)
        print('sensor_range: ', self.sensor_range)

    def gridToCenteredObservation(self, agent_pos):
        """ agent-centered view"""
        assert(agent_pos.ndim == 1)
        observation = np.zeros((2 * self.sensor_range[0] + 1, 2 * self.sensor_range[1] + 1, 3), dtype='uint8')
        for x in xrange(agent_pos[0] - self.sensor_range[0], agent_pos[0] + self.sensor_range[0] + 1):
            for y in xrange(agent_pos[1] - self.sensor_range[1], agent_pos[1] + self.sensor_range[1] + 1):
                xx = x - agent_pos[0] + self.sensor_range[0]
                yy = y - agent_pos[1] + self.sensor_range[1]
                if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1] or self.frame[x][y] == 2:
                    observation[xx][yy] = [0, 0, 0]
                elif self.frame[x][y] == 0:
                    observation[xx][yy] = [255, 255, 255]
                elif self.frame[x][y] == 1:
                    observation[xx][yy] = [150, 150, 150]
                else:
                    observation[xx][yy] = [255, 0, 0]
        return observation

    def render(self):
        """ agent-centered view"""
        img = np.zeros((self.frame.shape[0], self.frame.shape[1], 3), dtype = 'uint8')
        img[self.frame == 0] = [255, 255, 255]
        img[self.frame == 1] = [150, 150, 150]
        img[self.frame == 2] = [0, 0, 0]
        img[self.frame == 3] = [255, 0, 0]
        plt.imshow(img)
        plt.show()
        return img

    def resetAll(self):
        """ reset env and randomize obstacles
        """
        # 0: empty grid, 1: visited grid, 2: obstacle/boundary, 3: agent
        self.grid = np.zeros((self.grid_size[0], self.grid_size[1]), dtype='uint8')
        self.agent_pos = np.zeros((self.num_agents, 2), dtype='int32')
        self.num_obstacles = np.random.randint(self.obstacle_numrange[0], self.obstacle_numrange[1] + 1)
        self.obstacle_pos = np.zeros((self.num_obstacles, 2), dtype='int32')
        self.obstacle_size = np.zeros((self.num_obstacles, 2), dtype='int32')
        self.observations = np.zeros((self.num_agents, 2 * self.sensor_range[0] + 1, 2 * self.sensor_range[1] + 1, 3), dtype='uint8')
        self.unvisited_count = 0
        self.resetted = True
        for i in xrange(self.num_obstacles):
            # randomize size and pos of each obstacle
            w = np.random.randint(self.obstacle_sizerange[0], self.obstacle_sizerange[1] + 1)
            h = np.random.randint(self.obstacle_sizerange[0], self.obstacle_sizerange[1] + 1)
            x = np.random.randint(0, self.grid_size[0] - w + 1) 
            y = np.random.randint(0, self.grid_size[1] - h + 1)
            self.obstacle_pos[i][0] = x 
            self.obstacle_pos[i][1] = y
            self.obstacle_size[i][0] = w
            self.obstacle_size[i][1] = h

            # added obstacle to grid
            self.grid[x:(x + w - 1), y:(y + h - 1)] = 2

        self.frame = np.copy(self.grid)

        for i in xrange(self.num_agents):
            flag = True
            x = -1
            y = -1
            while (flag):
                x = np.random.randint(0, self.grid_size[0])
                y = np.random.randint(0, self.grid_size[1])
                if (self.frame[x][y] <= 1):
                    flag = False

            self.agent_pos[i][0] = x
            self.agent_pos[i][1] = y
            self.frame[x][y] = 3

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToCenteredObservation(self.agent_pos[i])

        for x in xrange(self.grid_size[0]):
            for y in xrange(self.grid_size[1]):
                if self.frame[x][y] == 0:
                    self.unvisited_count += 1

        return self.observations


    def reset(self):
        """ reset visited grid and agent positions only 
        """
        # MUST call resetAll first!
        assert(self.resetted)
        self.frame = np.copy(self.grid)
        self.unvisited_count = 0
        for i in xrange(self.num_agents):
            flag = True
            x = -1
            y = -1
            while (flag):
                x = np.random.randint(0, self.grid_size[0])
                y = np.random.randint(0, self.grid_size[1])
                if (self.frame[x][y] <= 1):
                    flag = False

            self.agent_pos[i][0] = x
            self.agent_pos[i][1] = y
            self.frame[x][y] = 3

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToCenteredObservation(self.agent_pos[i])

        for x in xrange(self.grid_size[0]):
            for y in xrange(self.grid_size[1]):
                if self.frame[x][y] == 0:
                    self.unvisited_count += 1

        return self.observations

    def step(self, actions):
        """
            Reward: -5: collision; 1: unvisited; 0: visited
        """
        assert(len(actions) == self.num_agents and self.resetted)
        order = np.random.permutation(self.num_agents)
        # hashtable = np.zeros_like(self.frame)
        # next_pos = np.zeros_like(self.agent_pos)

        rewards = np.zeros((self.num_agents,), dtype='float64')
        terminals = False
        for agentid in order:
            x = self.agent_pos[agentid][0] + self.directions[actions[agentid]]
            y = self.agent_pos[agentid][1] + self.directions[actions[agentid] + 1]
            if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1] or self.frame[x][y] >= 2:
                rewards[agentid] = -5
            elif self.frame[x][y] == 1:
                self.frame[self.agent_pos[agentid][0]][self.agent_pos[agentid][1]] = 1
                self.frame[x][y] = 3
                self.agent_pos[agentid][0] = x
                self.agent_pos[agentid][1] = y
                rewards[agentid] = 0
            else:
                self.frame[self.agent_pos[agentid][0]][self.agent_pos[agentid][1]] = 1
                self.frame[x][y] = 3
                self.agent_pos[agentid][0] = x
                self.agent_pos[agentid][1] = y
                self.unvisited_count -= 1
                rewards[agentid] = 1

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToCenteredObservation(self.agent_pos[i])
            
        if self.unvisited_count == 0:
            terminals = True

        return self.observations, rewards, terminals, self.unvisited_count

