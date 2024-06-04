# IMPORTS
import numpy as np
import random as random
import pulp
from grid import State
# from agent import Agent

pulp.LpSolverDefault.msg = 0

# GLOBAL VARIABLES

# how the grid is made:
BOARD_ROWS = 4
BOARD_COLS = 5
# number of iterations for the training and the testing phases
N_TRAIN = 500000
N_TEST = 100000
# in which states there could happen a win or a lose
POTENTIAL_LOSE_STATE_1 = (0, 1)
POTENTIAL_LOSE_STATE_2 = (0, 2)
POTENTIAL_WIN_STATE_1 = (4, 2)
POTENTIAL_WIN_STATE_2 = (4, 1) 
# initial positions of the two players
START_A = (3, 1)
START_B = (1, 2)


class Challenger:

    def availActions(self, state, opponent_flag = False):
        if state == (0, 0):
            action_set = ['S', 'E', 'stand']
        elif state == (4, 0):
            action_set = ['S', 'W', 'stand']
        elif state == (0, 3):
            action_set = ['N', 'E', 'stand']
        elif state == (4, 3):
            action_set = ['N', 'W', 'stand']
        elif state == (1, 0) or state == (2, 0) or state == (3, 0):
            action_set = ['S', 'E', 'W', 'stand']
        elif state == (1, 3) or state == (2, 3) or state == (3, 3):
            action_set = ['N', 'E', 'W', 'stand']
        elif (state == (4, 2) or state == (4, 1)) and opponent_flag==True:
            action_set = ['N', 'S', 'W', 'stand']
        elif (state == (0, 2) or state == (0, 1)) and opponent_flag==False:
            action_set = ['N', 'S', 'E', 'stand']
        else:
            action_set = ['N', 'S', 'W', 'E', 'stand']
        return action_set
    
    def __init__(self, grid, opponent_flag = False):
        self.grid = grid  # Defines the game field in which the agent and the opponent act
        self.isEnd = self.grid.isEnd  # If the game has ended, this attribute reflects that status
        print("--------------------------")
        print(f"Initial positions: {self.grid.state}")

        # Parameters for learning and exploration
        self.lr = 1.0  # Learning rate (alpha)
        self.exp_rate = 0.2  # Exploration rate (epsilon)
        self.gamma = 0.9  # Discount factor
        self.decay = 0.9999954  # Learning rate decay, ensuring lr reduces to 0.01 by the 1,000,000th iteration

        # Initializations for Q-values, V-values, and policy (pi)
        self.Q_values = {}  # Q-values: nested dictionaries for each triplet (s, a, o)
        self.V_values = {}  # V-values: dictionary for each state

        self.availActAgent = {}
        self.availActOpp = {}

        # Initialize Q-values, V-values, and policy for each possible state and action
        for i in range(BOARD_COLS):
            for j in range(BOARD_ROWS):
                for k in range(BOARD_COLS):
                    for l in range(BOARD_ROWS):  # Iterate over all possible positions
                        current_pos = str([(i, j), (k, l)])
                        self.V_values[current_pos] = 1
                        self.Q_values[current_pos] = {}
                        
                        if opponent_flag == True:
                            self.availActAgent[current_pos] = self.availActions((i, j), True)
                            self.availActOpp[current_pos] = self.availActions((k, l))
                        else:
                            self.availActAgent[current_pos] = self.availActions((k, l))
                            self.availActOpp[current_pos] = self.availActions((i, j), True)
                        
                        for a in self.availActAgent[current_pos]:  # Iterate over all actions
                            self.Q_values[current_pos][a] = 1

############################################# 

    def qLearningAction(self, action_b, position, new_position):
        """
        Updates Q-values, policy, and V-values based on the minimax algorithm after executing actions by both players.
        Args:
            - action_a (str): Action performed by player A.
            - action_b (str): Action performed by player B.
        """

        # Update Q-value for the current action pair based on the reward and the V-value of the new position
        old_Q = self.Q_values[position][action_b]
        self.Q_values[position][action_b] = (1 - self.lr) * old_Q + self.lr * (- self.grid.reward + self.gamma * self.V_values[new_position])

        # Update V-values for the current position
        self.V_values[position] = np.max([self.Q_values[position][a] for a in self.availActAgent[position]])

        # Update learning rate using decay factor
        self.lr = self.lr * self.decay

    def chooseqLearningAction(self):
        """
        Chooses an action using the minimax policy with an epsilon-greedy approach.
        Returns:
            - str: The chosen action.
        """
        # Initialize variable to store the chosen action
        action = ""

        # With probability exp_rate, choose a completely random action
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.availActAgent[str(self.grid.state)])

        # Otherwise, choose an action based on the minimax policy
        else:
            action = max(self.Q_values[str(self.grid.state)], key=self.Q_values[str(self.grid.state)].get)
        
        # Return the chosen action
        return action

