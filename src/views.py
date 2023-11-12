import matplotlib.pyplot as plt
from numpy import round as nround
from numpy import arange
from _checks import ArgumentIncorrectValueError


def show_distribution(data, title, label='Distribution', show=True) -> None:
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


def show_distributions(data1, data2, title, label1="First distribution", 
                       label2="Second distribution", show=True) -> None:
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
