#File for final testing
from mc import *

TPM = np.array([[1/3, 1/3, 1/3], 
                [1/3, 1/3, 1/3],
                [1/3, 1/3, 1/3]])
v0 = np.array([1.0, 0.0, 0.0])
states = np.array(['a', 'b', 'c'])

mc = MarkovChain(v0, TPM)
mc.set_states(states)
mc.set_initial_probabilities(np.array([0.5, 0.3, 0.2]))

print("Init probs:", mc.get_initial_probabilities(), sep="\n")
print("Transition matrix:", mc.get_transition_matrix(), sep="\n")
print("States:", mc.get_states(), sep="\n")