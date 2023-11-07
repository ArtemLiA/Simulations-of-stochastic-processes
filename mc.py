import numpy as np
from _checks import *


class MarkovChain:
    def __init__(self, initial_probabilities, transition_matrix, states = None) -> None:
        is_square_matrix(transition_matrix)
        is_stochastic_vector(initial_probabilities)
        is_stochastic_matrix(transition_matrix)
        is_agreed(initial_probabilities, transition_matrix)
        if states is not None:
            is_agreed(states, transition_matrix)
        
        self._states = np.arange(initial_probabilities.shape[0]) + 1 if states is None else states
        self._v0 = initial_probabilities
        self._P = transition_matrix

    def _simulate_trajectory(self, t) -> None:
        raise NotImplemented

    def _simulate_value(self, t) -> None:
        raise NotImplemented
    
    def simulate_trajectories(self, t, n_simulations = None) -> None:
        raise NotImplemented

    def simulate_values(self, t, n_simulations = None) -> None:
        raise NotImplemented

    def empirical_distribution(self, t, n_simulations = 100) -> None:
        raise NotImplemented

    def theoretical_distribution(self, t) -> None:
        raise NotImplemented

    def get_states(self):
        return np.copy(self._states)
    
    def get_transition_matrix(self):
        return np.copy(self._P)
    
    def get_initial_probabilities(self):
        return np.copy(self._v0)
    
    def set_states(self):
        raise NotImplemented
    
    def set_transition_matrix(self):
        raise NotImplemented
    
    def set_initial_probabilities(self, new_init_prob):
        is_stochastic_vector(new_init_prob)
        is_agreed(new_init_prob, self._P)
        self._v0 = new_init_prob
    