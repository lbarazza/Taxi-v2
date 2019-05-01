import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.03, gamma=.99):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.i = 1

    def action_prob(self, state, action):
        if action == np.argmax(self.Q[state]):
            return 1-self.epsilon + self.epsilon/self.nA
        else:
            return self.epsilon/self.nA
    
    def select_action(self, state, greedy=False): 
        if (np.random.random_sample() >= self.epsilon) or greedy:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(self.nA)
        return action

    def step(self, state, action, reward, next_state, done):
    
        #### Q-learning (for Sarsa set greedy to False)
        #next_action = self.select_action(next_state, greedy=True)
        #self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        
        #### Expected Sarsa
        Q_next_action = 0
        for next_action in range(0, self.nA):
            Q_next_action += self.Q[next_state][next_action] * self.action_prob(next_state, next_action)
        self.Q[state][action] += self.alpha * (reward + self.gamma * Q_next_action - self.Q[state][action])
        
        if done:
            self.i += 1
            self.epsilon = 1.0/(self.i)
