import os, sys, pickle

import numpy as np

def value_iteration(mdp, eps = 0.001):
    """
        A Fast vectorized value iteration for markov decision processes.
    """
    states, actions = list(mdp.states), list(mdp.actions)
    R, T, gamma = -1.0*np.array(mdp.costs).astype('float32'), np.array(mdp.transitions).astype('float32'), mdp.gamma

    V = np.zeros((len(states))).astype('float32').reshape(-1,1)
    Q = np.zeros((len(states), len(actions))).astype('float32')

    dim_array = np.ones((1, T.ndim), int).ravel()
    dim_array[2] = -1

    while True:
        Q = R + gamma*( np.sum( T * V.reshape(dim_array), axis = 2) )
        tmp = np.amax(Q, axis = 1)
        if np.max( abs(tmp - V) ) < eps:
            V = tmp
            break
        V = tmp
    V *= -1.0

    v = {s: V[s] for s in range(len(states))}
    pi = {s: actions[np.argmax(Q[s])] for s in range(len(states))}
    state_map = {state: s for s, state in enumerate(states)}

    results = {
        'V': V,
        'pi': pi,
        'state map': state_map,
        'Q' : Q
    }

    return results