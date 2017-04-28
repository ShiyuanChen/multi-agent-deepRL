import numpy as np
import matplotlib
# matplotlib.use('GTKAgg') 
import matplotlib.pyplot as plt

# Reward: collision=-5; visit new cell=1; else=0
REWARD_COLLISION = -1
REWARD_VISIT = 1
REWARD_NONE = 0

# Grid label: unvisited cell=0; visited cell=1; obstacle/boundary=2; occupied by agent=3
GRID_EMPTY = 0
GRID_VISITED = 1
GRID_OBSTACLE = 2
GRID_AGENT = 3

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
        plt.ion()

    def printInfo(self):
        print('grid_size: ', self.grid_size)
        print('num_actions: ', self.num_actions)
        print('num_agents: ', self.num_agents)
        print('obstacle_sizerange: ', self.obstacle_sizerange)
        print('sensor_range: ', self.sensor_range)

    def gridToCenteredObservation(self, agent_pos):
        """ agent-centered view
        """
        assert(agent_pos.ndim == 1)
        observation = np.zeros((2 * self.sensor_range[0] + 1, 2 * self.sensor_range[1] + 1, 3), dtype='uint8')
        for x in xrange(agent_pos[0] - self.sensor_range[0], agent_pos[0] + self.sensor_range[0] + 1):
            for y in xrange(agent_pos[1] - self.sensor_range[1], agent_pos[1] + self.sensor_range[1] + 1):
                xx = x - agent_pos[0] + self.sensor_range[0]
                yy = y - agent_pos[1] + self.sensor_range[1]
                if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1] or self.frame[x][y] == GRID_OBSTACLE:
                    observation[xx][yy] = [0, 0, 0]
                elif self.frame[x][y] == GRID_EMPTY:
                    observation[xx][yy] = [255, 255, 255]
                elif self.frame[x][y] == GRID_VISITED:
                    observation[xx][yy] = [170, 170, 170]
                else:
                    observation[xx][yy] = [255, 0, 0]
        return observation

    def gridToGlobalObservation(self):
        """ global view
        """
        observation = np.zeros((self.frame.shape[0], self.frame.shape[1], 3), dtype = 'uint8')
        for x in xrange(self.frame.shape[0]):
            for y in xrange(self.frame.shape[1]):
                if self.frame[x][y] == GRID_OBSTACLE:
                    observation[x][y] = [0, 0, 0]
                elif self.frame[x][y] == GRID_EMPTY:
                    observation[x][y] = [255, 255, 255]
                elif self.frame[x][y] == GRID_VISITED:
                    observation[x][y] = [170, 170, 170]
                else:
                    observation[x][y] = [255, 0, 0]
        return observation

    def render(self):
        """ render the whole grid
            TODO: non-blocking plot
        """
        plt.cla()
        img = np.zeros((self.frame.shape[0], self.frame.shape[1], 3), dtype = 'uint8')
        img[self.frame == GRID_EMPTY] = [255, 255, 255]
        img[self.frame == GRID_VISITED] = [170, 170, 170]
        # img[self.frame == GRID_OBSTACLE] = [0, 0, 0]
        img[self.frame == GRID_AGENT] = [255, 0, 0]

        plt.imshow(img)
        plt.draw()
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
        self.observations = np.zeros((self.num_agents, self.grid_size[0], self.grid_size[1], 3), dtype='uint8')
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
            self.grid[x:(x + w), y:(y + h)] = GRID_OBSTACLE

        self.frame = np.copy(self.grid)

        for i in xrange(self.num_agents):
            flag = True
            x = -1
            y = -1
            while (flag):
                x = np.random.randint(0, self.grid_size[0])
                y = np.random.randint(0, self.grid_size[1])
                if (self.frame[x][y] != GRID_OBSTACLE and self.frame[x][y] != GRID_AGENT):
                    flag = False

            self.agent_pos[i][0] = x
            self.agent_pos[i][1] = y
            self.frame[x][y] = GRID_AGENT

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToGlobalObservation()

        for x in xrange(self.grid_size[0]):
            for y in xrange(self.grid_size[1]):
                if self.frame[x][y] == GRID_EMPTY:
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
                if (self.frame[x][y] != GRID_OBSTACLE and self.frame[x][y] != GRID_AGENT):
                    flag = False

            self.agent_pos[i][0] = x
            self.agent_pos[i][1] = y
            self.frame[x][y] = GRID_AGENT

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToGlobalObservation()

        for x in xrange(self.grid_size[0]):
            for y in xrange(self.grid_size[1]):
                if self.frame[x][y] == GRID_EMPTY:
                    self.unvisited_count += 1

        return self.observations

    def step(self, actions):
        """ actions: list of action values

            Return:
            observations
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
            if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1] or self.frame[x][y] >= GRID_OBSTACLE:
                rewards[agentid] = REWARD_COLLISION
            elif self.frame[x][y] == GRID_VISITED:
                self.frame[self.agent_pos[agentid][0]][self.agent_pos[agentid][1]] = GRID_VISITED
                self.frame[x][y] = GRID_AGENT
                self.agent_pos[agentid][0] = x
                self.agent_pos[agentid][1] = y
                rewards[agentid] = REWARD_NONE
            else:
                self.frame[self.agent_pos[agentid][0]][self.agent_pos[agentid][1]] = GRID_VISITED
                self.frame[x][y] = GRID_AGENT
                self.agent_pos[agentid][0] = x
                self.agent_pos[agentid][1] = y
                self.unvisited_count -= 1
                rewards[agentid] = REWARD_VISIT

        for i in xrange(self.num_agents):
            self.observations[i] = self.gridToGlobalObservation()
            
        if self.unvisited_count == 0:
            terminals = True

        return self.observations, rewards, terminals, self.unvisited_count

