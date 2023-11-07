#File for final testing
from mc import *

TPM = np.array([[0.5, 0.5], [0.3, 0.7]])
v0 = np.array([0.5, 0.5])
x = np.array([4, 5])

mc = MarkovChain(v0, TPM, x)

print(mc.get_transition_matrix())
print(mc.get_initial_probabilities())
print(mc.get_states())
