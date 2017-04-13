"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl import utils
from deeprl.core import Sample, Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """
    def __init__(self, history_length=1):
        self.history_length = history_length
        self.state_history = None

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        num_agents = state.shape[0]
        if self.state_history is None:
            self.state_history = np.zeros(state.shape + (self.history_length,), dtype='uint8')
            for i in xrange(num_agents):
                self.state_history[i] = np.stack([np.zeros_like(state[i])] * (self.history_length - 1) + [state[i]], axis=2)
        else:
            for i in xrange(num_agents):
                self.state_history[i] = np.append(self.state_history[i][:,:,1:], np.expand_dims(state[i], 2), axis=2)
            
        return self.state_history

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.state_history = None

    def get_config(self):
        return {'history_length': self.history_length}


class GridPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.input_shape = new_size

    def process_state_for_memory(self, frame):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        num_agents = frame.shape[0]
        processed_frame = np.zeros((num_agents,) + self.input_shape, dtype='uint8')
        for i in xrange(num_agents):
            img = Image.fromarray(frame[i])
            processed_frame[i] = np.array(img.resize(self.input_shape).convert('L')).astype('uint8')
        return processed_frame

    def process_state_for_network(self, frame):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        num_agents = frame.shape[0]
        processed_frame = np.zeros((num_agents,) + self.input_shape, dtype='float32')
        for i in xrange(num_agents):
            img = Image.fromarray(frame[i])
            processed_frame[i] = np.array(img.resize(self.input_shape).convert('L')).astype('float32') / 255.
        return processed_frame

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        # float_sample = []
        # for sample in samples:
        #     float_sample.append(Sample(sample.state.astype('float32') / 255., sample.action, sample.reward, 
        #                                sample.next_state.astype('float32') / 255., sample.is_terminal))
        # return float_sample
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for sample in samples:
            num_agents = sample.state.shape[0]
            assert num_agents == sample.action.shape[0]
            for i in xrange(num_agents):    
                states.append(sample.state[i].astype('float32') / 255.)
                actions.append(sample.action[i])
                rewards.append(sample.reward[i])
                next_states.append(sample.next_state[i].astype('float32') / 255.)
                terminals.append(0. if sample.is_terminal else 1.)

        return np.array(states, dtype='float32'), np.array(actions), np.array(rewards), np.array(next_states, dtype='float32'), np.array(terminals)

    def process_batch_state_for_network(self, batch_states):
        return batch_states.astype('float32') / 255.

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1., 1.)

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        pass

class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.history = preprocessors[0]
        self.atari = preprocessors[1]

    def process_state_for_memory(self, state):
        """
        Return preprocessed frames with history_length
        """
        state = self.atari.process_state_for_memory(state)
        return self.history.process_state_for_network(state)

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return self.atari.process_batch(samples)

    def process_batch_state_for_network(self, batch_states):
        return self.atari.process_batch_state_for_network(batch_states)

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        # return self.atari.process_reward(reward)
        return reward

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.atari.reset()
        self.history.reset()