"""Main DQN agent."""
import os
from deeprl.policy import UniformRandomPolicy, GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl.objectives import huber_loss, mean_huber_loss
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Lambda, Input, Layer, Dense
import numpy as np
from gym import wrappers

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 num_actions,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 output_dir,
                 model_save_freq=1000000,
                 max_grad=1.):

        self.model = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.policy.set_agent(agent=self)
        # policy for evaluation
        self.eval_policy = GreedyEpsilonPolicy(epsilon=.05)

        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.max_grad = max_grad
        self.output_dir = output_dir
        self.model_save_freq = model_save_freq
        self.video_save_freq = 50

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        # create target model
        config = self.model.get_config()
        self.target_model = Sequential.from_config(config)
        self.target_model.set_weights(self.model.get_weights())
        print(self.target_model.summary())
        # compile both models. Trainable model will have separate loss later
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')
        
        # mask for output q values: one for current action and zero for others
        y_mask = Input(name='y_mask', shape=(self.num_actions,))
        y_pred = self.model.output
        # mask layer for trainable model. Share weight from trainable model
        y_masked_pred = Lambda(lambda y: y[0] * y[1], output_shape=(self.num_actions,), name='output')([y_pred, y_mask])
        self.q_model = Model(inputs=[self.model.input, y_mask], outputs=y_masked_pred)
        print(self.q_model.summary())

        # huber loss function for trainable model
        losses = [lambda y_true, y_pred: K.sum(huber_loss(y_true, y_pred, self.max_grad), axis=-1)]
        def mean_q_value(y_true, y_pred):
            return K.mean(K.max(y_pred, axis=-1))
        metrics = [mean_q_value]
        self.q_model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # Convert to float32
        process_state = self.preprocessor.process_batch_state_for_network(state)
        # forward pass for prediction
        q_values = self.model.predict_on_batch(process_state)
        return q_values

    def select_action(self, state, isTraining, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """

        q_values = self.calc_q_values(state)
        if isTraining:
            return self.policy.select_action(q_values=q_values)
        else:
            return self.eval_policy.select_action(q_values=q_values)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        metrics = []
        # Skip update if burn-in or not at the step for update
        if self.step > self.num_burn_in and self.step % self.train_freq == 0:
            # minibatch forward pass
            samples = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, terminals = self.preprocessor.process_batch(samples)
            
            target_q_values = self.target_model.predict_on_batch(next_states)
            max_target_q_values = np.max(target_q_values, axis=1).flatten()

            sample_size = states.shape[0]
            target_g_values = np.zeros((sample_size, self.num_actions))
            masks = np.zeros((sample_size, self.num_actions))

            # target reward
            Rs = rewards + terminals * self.gamma * max_target_q_values 

            for idx, (action, target_g, mask, R) in enumerate(zip(actions, target_g_values, masks, Rs)):
                target_g[action] = R
                mask[action] = 1.
            targets = np.array(target_g_values).astype('float32')
            masks = np.array(masks).astype('float32')

            # backprop to update
            metrics = self.q_model.train_on_batch([states, masks], targets)
            self.loss.append(metrics)

        # update weights of target model from trainable model
        if self.target_update_freq >= 1 and self.step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return metrics


    def video_callable(self, episode_id):
        """Interval to record video using Gym monitor
        """
        return episode_id % self.video_save_freq == 0


    def fit(self, env, num_iterations, callbacks=None, max_episode_length=None, episode_log_freq=20):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        self.step = 0
        episode_idx = 0
        state = None
        episode_reward = None
        need_eval = False

        [model_dir, video_dir] = self.create_output_dirs(self.output_dir, ['models','videos'])

        # Monitor
        # env = wrappers.Monitor(env, video_dir, video_callable=self.video_callable)

        episode_rewards = []
        episode_length = []
        last_episode_length = 0
        self.loss = []

        for callback in callbacks:
            callback.set_model(self)       

        while self.step < num_iterations:
            # start a new episode
            if state is None:
                episode_reward = 0.
                self.preprocessor.reset()
                state = env.resetAll()
                state = self.preprocessor.process_state_for_memory(state)

            # run a single step.
            # env.render()
            action = self.select_action(state, isTraining=True)
            # assert action.shape[0] == state.shape[0]
            next_state, unclipped_reward, done, info = env.step(action)
            next_state = self.preprocessor.process_state_for_memory(next_state)
            reward = self.preprocessor.process_reward(unclipped_reward)

            if self.step - last_episode_length >= max_episode_length - 1:
                done = True

            # store transition to memory
            self.memory.append(state, action, reward, next_state, done)

            # update weight
            metrics = self.update_policy()
            for r in unclipped_reward:
                episode_reward += r

            self.step += 1
            # print str(self.step) + ': '
            # print reward
            # print episode_reward
            # save models
            if self.step % self.model_save_freq == 0:
                self.save_weights(os.path.join(model_dir, 'model-step-{}'.format(self.step)), overwrite=False)

            if done: # end of episode
                episode_rewards.append(episode_reward)
                episode_length.append(self.step - last_episode_length)
                last_episode_length = self.step

                episode_idx += 1
                print('---------------------------------------------')
                print('episode #: {}    Iteration #: {}    episode_reward: {}    epsilon: {}'.format(episode_idx, self.step, episode_reward, self.policy.value))
                if episode_idx % episode_log_freq == 0:
                    episode_rewards = np.array(episode_rewards)
                    self.loss = np.array(self.loss)
                    episode_length = np.array(episode_length)
                    episode_logs = {}
                    if self.step > self.num_burn_in + 1:
                        metrics_mean = np.mean(self.loss, axis=0)
                        metrics_std = np.std(self.loss, axis=0)
                        episode_logs = {
                            'episode_reward_mean': np.mean(episode_rewards),
                            'episode_reward_std': np.std(episode_rewards),
                            'episode_length_mean': np.mean(episode_length),
                            'episode_length_std': np.std(episode_length),
                            'loss_mean': metrics_mean[0],
                            'loss_std': metrics_std[0],
                            'q_values_mean': metrics_mean[1],
                            'q_values_std': metrics_std[1],
                            'episode': np.array(episode_idx),
                        }
                        
                        print('episode_reward: {}    episode_length: {}'.format(
                               episode_logs['episode_reward_mean'], episode_logs['episode_length_mean']))
                        print('mean_q_values:{}    loss: {}'.format(
                               episode_logs['q_values_mean'], episode_logs['loss_mean']))
                    else:
                        episode_logs = {
                            'episode_reward_mean': np.mean(episode_rewards),
                            'episode_reward_std': np.std(episode_rewards),
                            'episode_length_mean': np.mean(episode_length),
                            'episode_length_std': np.std(episode_length),
                            'episode': np.array(episode_idx),
                        }
                        print('episode_reward: {}    episode_length: {}'.format(
                               episode_logs['episode_reward_mean'], episode_logs['episode_length_mean']))
                    
                    for callback in callbacks:
                        callback.on_epoch_end(episode_idx, episode_logs)

                    episode_rewards = []
                    episode_length = []
                    self.loss = []

                print('')
                state = None
                episode_reward = None            
            else: # continue current episode
                state = next_state

        # env.close()
        for callback in callbacks:
            callback.on_train_end('end')

        # save final weights and evaluate 100 episodes
        self.save_weights(os.path.join(model_dir, 'model-step-{}'.format(self.step)), overwrite=True)
        self.evaluate(env, 100)
        

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        [video_dir] = self.create_output_dirs(self.output_dir, ['videos_test'])
        # env = wrappers.Monitor(env, video_dir, video_callable=self.video_callable)
        test_step = 0
        episode_rewards = []
        episode_idx = 0
        print('---------------Start Evaluating!-----------------')
        while episode_idx < num_episodes:
            
            episode_reward = 0.
            self.preprocessor.reset()
            state = env.resetAll()
            state = self.preprocessor.process_state_for_memory(state)
        
            done = False
            while not done:
                # run a single step.
                action = self.select_action(state, isTraining=False)
                state, reward, done, _ = env.step(action)
                state = self.preprocessor.process_state_for_memory(state)

                episode_reward += reward
                test_step += 1

            episode_idx += 1
            episode_rewards.append(episode_reward)
            print('---------------------------------------------')
            print('episode #: {}    Iteration #: {}    episode_reward: {}    epsilon: {}'.format(episode_idx, test_step, episode_reward, self.eval_policy.eps))

        # env.close()

        episode_rewards = np.array(episode_rewards)
        print('---------------End Evaluating!-----------------')
        print("Average Total Reward: {}    std: {}".format(np.mean(episode_rewards), np.std(episode_rewards)))


    def save_weights(self, filepath, overwrite=False):
        # save weights of a model to file
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        # load weights from file
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())


    def create_output_dirs(self, parent_dir, dirs):
        folder_paths = []
        for directory in dirs:
            folder_path = os.path.join(parent_dir, directory)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            folder_paths.append(folder_path)
        return folder_paths
