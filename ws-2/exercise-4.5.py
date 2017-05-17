"""
Author: Amol Kapoor
Description: Implementation of Jack's Car Rental problem from Sutton, exercise
4.5.
"""

import itertools
import math
import numpy as np
from scipy.stats import poisson

# System vars
RENT_PRICE = 10
COST_TO_MOVE_CARS = 2
MAX_CARS = 20
MAX_CARS_MOVED = 5
ACTIONS_AVAILABLE = MAX_CARS_MOVED*2 + 1
GAMMA = 0.9
LAMBDAS = {
    'req_one': 3,
    'req_two': 4,
    'ret_one': 3,
    'ret_two': 2
}
DEBUG_STEPS = 100

def get_reward(r_one, r_two, action):
    """
    Gets the reward based on rent prices for cars rented.
    """
    return RENT_PRICE * (r_one + r_two) - COST_TO_MOVE_CARS * (
        action - MAX_CARS_MOVED)

def run_policy(policy, state):
    """
    Returns the recommend action for a given state.
    """
    return policy[state[0]][state[1]]

def print_policy(policy):
    """
    Prints the policy.
    """
    for row in range(policy.shape[0]):
        for col in range(policy.shape[1]):
            print int(abs(policy[row][col])),
        print
    print '---------------------------------------------------'

def print_q(vals):
    """
    Prints the q values.
    """
    for row in range(vals.shape[0]):
        for col in range(vals.shape[1]):
            print int(abs(np.max(vals[row][col]))),
        print
    print '---------------------------------------------------'

def probs(cars, req_mu, ret_mu):
    dist = {}
    for reqs in xrange(0, cars + 1):
        max_ret = MAX_CARS - (cars - reqs)
        for rets in xrange(0, max_ret + 1):
            p_req = poisson.pmf(req_mu, reqs)
            p_ret = poisson.pmf(ret_mu, rets)
            if math.isnan(p_req) or math.isnan(p_req):
                p_req = 0
                p_ret = 0
            dist[(reqs, rets)] = p_req * p_ret
    return dist

def _update_q(values, policy, state):
    """
    Actually updates a specific q.
    """
    carsl1, carsl2, action = state
    carsl1 -= action
    carsl2 += action
    l1_probs = probs(carsl1, LAMBDAS['req_one'], LAMBDAS['ret_one'])
    l2_probs = probs(carsl2, LAMBDAS['req_two'], LAMBDAS['ret_two'])

    total_reward = 0

    for (rq1, rt1), prob1 in l1_probs.iteritems():
        for (rq2, rt2), prob2 in l2_probs.iteritems():
            prob = prob1*prob2
            reward = get_reward(rq1, rq2, action)
            new_state = (carsl1 - rq1 + rt1, carsl2 - rq2 + rt2)
            total_reward += prob * (reward + GAMMA * values[state[0]][state[1]])
    return total_reward

def update_q(values, policy):
    """
    Updates state-action values by iterating through all combinations and
    simulating rewards. Does value iteration (i.e. single update per state)
    """
    cars_one, cars_two = values.shape
    new_values = np.copy(values)
    for i, j in itertools.product(xrange(cars_one), xrange(cars_two)):
        action = run_policy(policy, (i, j))
        new_values[i][j] = _update_q(values, policy, (i, j, action))
    return new_values

def _update_pi(policy, values, state):
    c1, c2 = state
    c1_actions = range(min(c1, MAX_CARS_MOVED) + 1)
    c2_actions = range(-min(c2, MAX_CARS_MOVED), 1)

    action_state_map = {}
    for action in c1_actions + c2_actions:
        if abs(c1 - action) > MAX_CARS or c2 + action > MAX_CARS:
            continue
        new_state_val = values[c1 - action][c2 + action]
        action_state_map[new_state_val] = action

    return action_state_map[np.max(action_state_map.keys())]


def update_pi(policy, values):
    """
    Greedily updates policy.
    """
    cars_one, cars_two = values.shape
    for i, j in itertools.product(xrange(cars_one), xrange(cars_two)):
        policy[i][j] = _update_pi(policy, values, (i, j))
    return policy


# Variables
# Stores reward for each state action pair
q = np.random.randint(0, 30, (MAX_CARS + 1, MAX_CARS + 1), np.int)
# Number of cars to move given a state
pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), np.int)

num_steps = 0
while True:
    # Q update
    q = update_q(q, pi)
    print_q(q)
    # Policy Update
    temp = np.copy(pi)
    pi = update_pi(pi, q)
    print_policy(pi)
    if np.array_equal(temp, pi):
        break

print print_policy(pi)
