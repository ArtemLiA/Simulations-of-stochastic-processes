import matplotlib.pyplot as plt

from numpy import round as nround
from numpy import arange
from numpy import linspace
from numpy import min as nmin
from numpy import max as nmax

from _checks import ArgumentIncorrectValueError
from mc import Simulation


def show_trajectory(sim: Simulation, title: str, show: bool = True):
    states, times = sim.states, sim.times
    step = 5 if nmax(times) - nmin(times) >= 9 else 5 
    
    x_ticks = linspace(nmin(times), nmax(times), step)
    y_ticks = y_ticks = arange(nmin(states)-2, nmax(states)+3)
    
    plt.scatter(times, states, s=50, c='m', label='Simulation states')
    plt.plot(times, states, 'c')
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if show: plt.show()


def show_distribution(data: dict, title: str, label: str = 'Distribution', show: bool = True) -> None:
    x = [str(item[0]) for item in data.items()]
    height = [nround(item[1], 3) for item in data.items()]
    color = ['red'] * len(x)
    
    rects = plt.bar(x, height, color=color, label=label)
    plt.bar_label(rects, padding=3)
    plt.xlabel('States')
    plt.ylabel('Probabilities')
    plt.ylim((0, 1))
    plt.title(title)
    plt.legend(loc='upper left')
    if show: plt.show()


def show_distributions(data1: dict, data2: dict, title: str, label1: str = "First distribution", 
                       label2: str = "Second distribution", show: bool = True) -> None:
    if list(data1.keys()) != list(data2.keys()):
        raise ArgumentIncorrectValueError("data1 and data2 have different keys.")
    
    states = [str(item[0]) for item in data1.items()]
    p1 = [nround(item[1], 3) for item in data1.items()]
    p2 = [nround(item[1], 3) for item in data2.items()]
        
    x = arange(len(states))
    width = 0.3
    multiplier = 0
    colors = {
        label1: ['red'] * len(states),
        label2: ['magenta'] * len(states)
    }
    
    full_data = {
        label1: p1,
        label2: p2
    }
    
    for label, p_list in full_data.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, p_list, width, label=label, color=colors[label])
        plt.bar_label(rects, padding=3)
        multiplier += 1
    
    plt.xlabel('States')
    plt.ylabel('Probabilities')
    plt.title(title)
    plt.xticks(x + width/2, states)
    plt.legend(loc='upper left', ncols=2)
    plt.ylim((0, 1))
    if show: plt.show()
