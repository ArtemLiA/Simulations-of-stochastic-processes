import numpy as np
import matplotlib.pyplot as plt
from _checks import *
from collections import namedtuple
from typing import Union


Simulation = namedtuple('Simulation', ['states', 'times'])
Simulation_series = namedtuple('Simulation_series', ['states', 'times'])


class MarkovChain:
    def __init__(self, 
                 initial_probabilities: np.array, 
                 transition_matrix: np.ndarray, 
                 states: Union[np.array, None] = None) -> None:
        is_square_matrix(transition_matrix)
        is_stochastic_vector(initial_probabilities)
        is_stochastic_matrix(transition_matrix)
        is_agreed(initial_probabilities, transition_matrix)
        if states is not None:
            is_agreed(states, transition_matrix)
        
        self._states = np.arange(initial_probabilities.shape[0]) + 1 if states is None else states
        self._v0 = initial_probabilities
        self._P = transition_matrix

    def _simulate_trajectory(self, t: int) -> Simulation:
        all_indeces = np.arange(self._states.shape[0])
        current_index = np.random.choice(a=all_indeces, p=self._v0)

        simulation_states = np.array(self._states[current_index])

        for time in range(1, t+1):
            p_list = self._P[current_index]
            current_index = np.random.choice(a=all_indeces, p=p_list)

            simulation_states = np.append(simulation_states, self._states[current_index])
        
        return simulation_states

    def _simulate_state(self, t: int):
        all_indeces = np.arange(self._states.shape[0])
        current_index = np.random.choice(a=all_indeces, p=self._v0)

        for time in range(1, t+1):
            p_list = self._P[current_index]
            current_index = np.random.choice(a=all_indeces, p=p_list)
        
        return self._states[current_index]
    
    def simulate_trajectories(self, t: int, n_simulations: Union[int, None] = None) -> Union[Simulation, Simulation_series]:
        if n_simulations is None:
            simulation_times = np.arange(0, t+1)
            simulation_states = self._simulate_trajectory(t)
            return Simulation(self._simulate_trajectory(t), simulation_times)
        
        simulations_times = np.arange(0, t+1)
        simulations_states = np.empty((0, t+1))

        for _ in range(n_simulations):
            sim_states = self._simulate_trajectory(t)
            simulations_states = np.append(simulations_states, sim_states.reshape(1, -1), axis=0)

        return Simulation_series(simulations_states, simulations_times)
        
    def simulate_states(self, t: int, n_simulations: Union[int, None] = None):
        if n_simulations is None:
            return self._simulate_state(t)
        
        simulations_states = np.array([], dtype=self._states.dtype)
        
        for _ in range(n_simulations):
            state = self._simulate_state(t)
            simulations_states = np.append(simulations_states, state)
        
        return simulations_states

    def empirical_distribution(self, t: int, n_simulations: int = 100) -> dict:
        simulations_states = self.simulate_states(t=t, n_simulations=n_simulations)
        distribution = dict.fromkeys(self._states, 0)
        
        for state in simulations_states:
            distribution[state] += 1
        
        return {state: count/n_simulations for state, count in distribution.items()}

    def theoretical_distribution(self, t: int) -> dict:
        probabilities = self._v0 @ np.linalg.matrix_power(self._P, t)
        return dict(zip(self._states, probabilities))

    def get_states(self) -> np.array:
        return np.copy(self._states)
    
    def get_transition_matrix(self) -> np.ndarray:
        return np.copy(self._P)
    
    def get_initial_probabilities(self) -> np.array:
        return np.copy(self._v0)
    
    def set_states(self, new_states: Union[np.array, None] = None) -> None:
        if new_states is not None:
            is_agreed(new_states, self._P)
        self._states = np.arange(self._v0.shape[0]) + 1 if new_states is None else new_states
    
    def set_transition_matrix(self, new_transition_matrix: np.ndarray) -> None:
        is_stochastic_matrix(new_transition_matrix)
        is_agreed(self._v0, new_transition_matrix)
        self._P = new_transition_matrix
    
    def set_initial_probabilities(self, new_inittial_probabilities: np.array) -> None:
        is_stochastic_vector(new_inittial_probabilities)
        is_agreed(new_inittial_probabilities, self._P)
        self._v0 = new_inittial_probabilities
    
    def show_empirical_distribution(self, t, n_simulations, color, label, show=True):
        colors = [color] * self._states.shape[0]
        data = self.empirical_distribution(t=t, n_simulations=n_simulations)
        
        plt.bar(list(data.keys()), list(data.values()), label=label, color=colors)
        plt.xlabel('States')
        plt.ylabel('Probabilities')
        if show:
            plt.legend()
            plt.show()
    
    
def create_histogram(data, colors, title='Sample cross-section distribution'):
    """
    Creating bar by distribution
    :param data: dict - data
    """
    plt.bar(list(data.keys()), list(data.values()), color=colors)
    plt.xlabel('States')
    plt.ylabel('Probabilities estimation')
    plt.title(title)
    plt.show()
