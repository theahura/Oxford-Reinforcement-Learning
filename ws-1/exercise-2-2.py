"""
Author: Amol Kapoor
Description: Code for exercises 2.2, reading assignment 1, in Sutton.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot(results):
    plt.plot(range(0, len(results[0])), np.mean(results, axis=0))
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.show()

def greedypolicy(epsilon, estimates):
    if np.random.random_sample() < epsilon:
        return np.random.randint(0, len(estimates))
    else:
        return np.argmax(estimates)

def softmaxpolicy(temp, estimates):
    probs = []
    for i in xrange(len(estimates)):
        num = np.exp(estimates[i]/temp)
        den = 0.0
        for j in range(len(estimates)):
            if j != i:
                den += np.exp(estimates[j]/temp)
        probs.append(num/den)
    probs = np.array(probs)
    probs /= probs.sum()
    return np.random.choice(range(len(estimates)), p=probs)

def learn_bandits(tasks=2000, bandit_mean=0.0, bandit_deviation=1.0,
                  bandit_count=10, bandit_reward=1.0, steps=1000, stepsize=0.1,
                  epsilon=0.1, temp=0.5):
    results = []
    for _ in xrange(0, tasks):
        task_results = []
        bandit_means = np.random.normal(bandit_mean, bandit_deviation,
                                        bandit_count)
        estimates = [0.0]*bandit_count
        for _ in xrange(0, steps):
            bandit_index = softmaxpolicy(epsilon, estimates)
            reward = np.random.normal(bandit_means[bandit_index],
                                      bandit_deviation)
            estimates[bandit_index] = estimates[bandit_index] + stepsize*(reward
                    - estimates[bandit_index])
            task_results.append(reward)
        results.append(task_results)
    plot(results)

learn_bandits()
