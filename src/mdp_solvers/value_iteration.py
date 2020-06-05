import os, sys, pickle

import numpy as np

def value_iteration(mdp, eps = 0.001):
    """
        A Fast vectorized value iteration for markov decision processes.
    """

    # Import the mdp states and actions as lists
    states, actions, gamma = list(mdp.states), list(mdp.actions), mdp.gamma

    # Import the mdp transitions and costs and convert to float32 np arrays.
    # Currently expecting transitions and rewards to be fully computed and cached. 
    R, T = *np.array(mdp.rewards).astype('float32'), np.array(mdp.transitions).astype('float32')

    V = np.zeros((len(states), 1)).astype('float32')
    Q = np.zeros((len(states), len(actions))).astype('float32')

    # Minor book-keeping to make Q value calculation easy.
    dim_array = np.ones((1, T.ndim), int).ravel()
    dim_array[2] = -1

    # Main value iteration loop. 
    while True:
        Q = R + gamma*( np.sum( T * V.reshape(dim_array), axis = 2) )
        tmp = np.amax(Q, axis = 1)

        # Check termination condition. 
        if np.max( abs(tmp - V) ) < eps:
            V = tmp
            break
        V = tmp

    pi = {s: actions[np.argmax(Q[s])] for s in range(len(states))}
    state_map = {state: s for s, state in enumerate(states)}

    results = {
        'V': V,
        'Q' : Q,
        'pi': pi,
        'state map': state_map
    }

    return results