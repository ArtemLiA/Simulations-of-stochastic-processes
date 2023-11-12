#File for final testing
from mc import *
from views import *

TPM = np.array([[1/3, 1/3, 1/3], 
                [1/3, 1/3, 1/3],
                [1/3, 1/3, 1/3]])
v0 = np.array([1.0, 0.0, 0.0])
states = np.array(['a', 'b', 'c'])

mc = MarkovChain(v0, TPM)
mc.set_initial_probabilities(np.array([0.5, 0.3, 0.2]))

print("Init probs:", mc.get_initial_probabilities(), sep="\n")
print("Transition matrix:", mc.get_transition_matrix(), sep="\n")
print("States:", mc.get_states(), sep="\n")

d1 = mc.empirical_distribution(5, 150)
d2 = mc.theoretical_distribution(5)
trj = mc.simulate_trajectories(10)

show_distributions(d1, d2, "Example title", "d1", "d2")
show_trajectory(trj, "Some title")

