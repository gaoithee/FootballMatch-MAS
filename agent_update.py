# IMPORTS
import numpy as np
import random as random
from grid import State
from scipy.optimize import linprog
import pulp
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


class Agent:

    def availActions(self, state, opponent = False):
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
        elif (state == (4, 2) or state == (4, 1)) and opponent==True:
            action_set = ['N', 'S', 'W', 'stand']
        elif (state == (0, 2) or state == (0, 1)) and opponent==False:
            action_set = ['N', 'S', 'E', 'stand']
        else:
            action_set = ['N', 'S', 'W', 'E', 'stand']
        return action_set
    
    def __init__(self, grid, opponent_flag = False):
        self.opponent = opponent_flag
        self.grid = grid  # Defines the game field in which the agent and the opponent act
        self.isEnd = self.grid.isEnd  # If the game has ended, this attribute reflects that status

        # Parameters for learning and exploration
        self.lr = 1.0  # Learning rate (alpha)
        self.exp_rate = 0.2  # Exploration rate (epsilon)
        self.gamma = 0.9  # Discount factor
        self.decay = 0.9999954  # Learning rate decay, ensuring lr reduces to 0.01 by the 1,000,000th iteration

        # Initializations for Q-values, V-values, and policy (pi)
        self.Q_values = {}  # Q-values: nested dictionaries for each triplet (s, a, o)
        self.V_values = {}  # V-values: dictionary for each state
        self.pi = {}  # Policy: nested dictionary for each pair (s, a)
        self.B = {} # Belief: also a nested dictionary for each pair (s, o)

        # Initialization of parameters needed to update `self.B`:
        self.history = {} # Dictionary counting the absolute frequency of each o in each s
        self.counter = {} # Overall counter to obtain the relative frequencies for each s

        # Which actions can be taken by the Agent and the Opponent in each state of the board
        self.availActAgent = {}
        self.availActOpp = {}

        # Initialize Q-values, V-values, and policy for each possible state and action
        for i in range(BOARD_COLS):
            for j in range(BOARD_ROWS):
                for k in range(BOARD_COLS):
                    for l in range(BOARD_ROWS):

                        current_pos = str([(i, j), (k, l)])
                        self.V_values[current_pos] = 1
                        self.Q_values[current_pos] = {}
                        self.pi[current_pos] = {}
                        self.B[current_pos] = {}
                        self.history[current_pos] = {}

                        if self.opponent == True:
                            self.availActAgent[current_pos] = self.availActions((i, j), True)
                            self.availActOpp[current_pos] = self.availActions((k, l))
                        else:
                            self.availActAgent[current_pos] = self.availActions((k, l))
                            self.availActOpp[current_pos] = self.availActions((i, j), True)
                        self.counter[current_pos] = len(self.availActOpp[current_pos])
                        
                        for a in self.availActAgent[current_pos]:  # Iterate over all actions
                            self.pi[current_pos][a] = 1 / len(self.availActAgent[current_pos])
                            self.Q_values[current_pos][a] = {}
                            for o in self.availActOpp[current_pos]:  # Iterate over opponent's actions
                                self.Q_values[current_pos][a][o] = 1  # Initialize Q-value as a nested dictionary

                        for o in self.availActOpp[current_pos]:
                            self.B[current_pos][o] = 1/len(self.availActOpp[current_pos]) 
                            self.history[current_pos][o] = 1

#############################################
    
    def minimaxPolicy(self):
        """
        Computes the minimax Q-learning policy for the current state using linear programming.
        Returns:
            - dict: Updated minimax policy for the current state.
        """
        position = str(self.grid.state)
        
        num_actions = len(self.availActAgent[position])
        num_opponent_actions = len(self.availActOpp[position])
        
        # Coefficients for the objective function (to maximize max_min_value)
        epsilon = 1e-6
        c = np.zeros(num_actions + 1)
        c[-1] = -1  # We want to maximize max_min_value, so we minimize its negative
        c[:-1] = epsilon

        # Inequality matrix and vector
        A = np.zeros((num_opponent_actions, num_actions + 1))
        b = np.zeros(num_opponent_actions)
        
        for j, o in enumerate(self.availActOpp[position]):
            for i, a in enumerate(self.availActAgent[position]):
                A[j, i] = -self.Q_values[position][a][o] 
            A[j, -1] = 1  # max_min_value term
        
        # Equality constraint matrix and vector
        A_eq = np.ones((1, num_actions + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1])
        
        # Bounds for the variables
        bounds = [(0, 1) for _ in range(num_actions)] + [(None, None)]
        
        # Solve the linear programming problem
        result = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        if not result.success:
            raise ValueError("Linear programming problem did not converge")
        
        # Extract the solution values for the policy and ensure they are non-negative
        solution = result.x[:-1]
        solution = np.maximum(solution, 0)

            
        # Normalize the solution so that the sum of the probabilities equals 1
        total = np.sum(solution)
        if total > 0:
            solution /= total
            
        # Correction for probabilities close to 1
        for i, val in enumerate(solution):
            if val >= 0.999:
                solution = np.zeros_like(solution)
                solution[i] = 1.0
                break
            
        # If no probability is close to 1, distribute the difference proportionally
        total = np.sum(solution)
        if not abs(total - 1.0) < 1e-5:
            if total > 1:
                excess = total - 1
                solution = [max(0, val - excess * (val / total)) for val in solution]
            else:
                deficit = 1 - total
                solution = [val + deficit * (val / total) for val in solution]
            
        # Ensure the sum is exactly 1 after correction
        total = np.sum(solution)
        if not abs(total - 1.0) < 1e-5:
            raise ValueError("Probabilities are not validly normalized")
            
        solution_dict = dict(zip(self.availActAgent[position], solution))
        self.pi[position] = solution_dict
        return self.pi

    def updateMinimaxValue(self, position):
        """
        Updates the minimax value for the given position based on the current policy and Q-values.
        Args:
            - position (tuple): The position/state for which the minimax value is to be updated.
        Returns:
            - dict: Updated V-values for the given position.
        """
        position = str(position)  # Convert the position to a string representation
        
        # Initialize a list to store the expected values for each action
        exp_values = []
        
        # Calculate the expected values for each action using the current policy and Q-values
        for o in self.availActOpp[position]:
            temp_exp_values = 0
            for a in self.availActAgent[position]:
                temp_exp_values += self.pi[position][a] * self.Q_values[position][a][o]
            exp_values.append(temp_exp_values)

        # Update the V-value for the given position to the minimum of the expected values
        self.V_values[position] = np.min(exp_values)
        return self.V_values

    def minimaxAction(self, action_a, action_b, position, new_position):
        """
        Updates Q-values, policy, and V-values based on the minimax algorithm after executing actions by both players.
        Args:
            - action_a (str): Action performed by player A.
            - action_b (str): Action performed by player B.
        """

        # Update Q-value for the current action pair based on the reward and the V-value of the new position
        old_Q = self.Q_values[position][action_b][action_a]
        if self.opponent == False:
            self.Q_values[position][action_b][action_a] =  (1 - self.lr) * old_Q + self.lr * (self.grid.reward + self.gamma * self.V_values[new_position])
        else:
            self.Q_values[position][action_b][action_a] =  (1 - self.lr) * old_Q + self.lr * (-self.grid.reward + self.gamma * self.V_values[new_position])

        # Update policy based on the minimax algorithm
        self.pi[position] =  self.minimaxPolicy()[position]

        # Update V-values for the current position
        self.V_values[position] = self.updateMinimaxValue(position)[position]

        # Update learning rate using decay factor
        self.lr = self.lr * self.decay
    
    def chooseMinimaxAction(self):
        """
        Chooses an action using the minimax policy with an epsilon-greedy approach.
        Returns:
            - str: The chosen action.
        """
        # Initialize variable to store the chosen action
        action = ""

        # if opponent_flag == True:
        #     temp = self.availActAgent[str(self.grid.state)]
        #     self.availActAgent[str(self.grid.state)] = self.availActOpp[str(self.grid.state)]
        #     self.availActOpp[str(self.grid.state)] = temp

        # With probability exp_rate, choose a completely random action
        if np.random.uniform(0, 1) <= self.exp_rate:
            # print(f"random action case. possible choices:", self.availActAgent[str(self.grid.state)])
            action = random.choice(self.availActAgent[str(self.grid.state)])
            # print(f"chosen: ", action)

        # Otherwise, choose an action based on the minimax policy
        else:
            # print(f"policy action case. possible choices:", self.availActAgent[str(self.grid.state)])
            # temp_dict = self.minimaxPolicy()[str(self.grid.state)]
            temp_dict = self.pi[str(self.grid.state)]
            acts = list(temp_dict.keys())
            probs = list(temp_dict.values())
            action =  np.random.choice(acts, p=probs)
            # action = self.pi[str(self.grid.state)]
            # print(f"chosen: ", action)
        # print(f"available for the agent: ", self.availActAgent[str(self.grid.state)])
        # print(f"available for the opponent: ", self.availActOpp[str(self.grid.state)])
        # Return the chosen action
        return action

#############################################

    def chooseRandomAction(self):
        chosen = random.choice(self.availActOpp[str(self.grid.state)])
        return chosen
    
#############################################

    def beliefBasedPolicy(self):
        """
        Computes the belief-based Q-learning policy for the current state based also on beliefs.
        Returns:
            - dict: Updated belief-based policy for the current state.
        """
        position = str(self.grid.state)
        
        num_actions = len(self.availActAgent[position])
        num_opponent_actions = len(self.availActOpp[position])
        
        # Coefficients for the objective function
        c = np.zeros(num_actions)
        for i, a in enumerate(self.availActAgent[position]):
            for o in self.availActOpp[position]:
                c[i] -= self.B[position][o] * self.Q_values[position][a][o]  # negate for maximization
        
        # Inequality constraints matrix and vector (none needed in this case)
        A_ub = None
        b_ub = None
        
        # Equality constraint matrix and vector
        A_eq = np.ones((1, num_actions))
        b_eq = np.array([1])
        
        # Bounds for the variables
        bounds = [(0, 0.9) for _ in range(num_actions)]
        
        # Solve the linear programming problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        if not result.success:
            raise ValueError("Solver did not find an optimal solution")
        
        # Extract the solution values for the policy and ensure they are non-negative
        solution = result.x
        solution = np.maximum(solution, 0)
        
        # Normalize the solution so that the sum of the probabilities equals 1
        total = np.sum(solution)
        if total > 0:
            solution /= total
        
        # Correction for probabilities close to 1
        for i, val in enumerate(solution):
            if val >= 0.999:
                solution = np.zeros_like(solution)
                solution[i] = 1.0
                break
        
        # If no probability is close to 1, distribute the difference proportionally
        total = np.sum(solution)
        if not abs(total - 1.0) < 1e-5:
            if total > 1:
                excess = total - 1
                solution = [max(0, val - excess * (val / total)) for val in solution]
            else:
                deficit = 1 - total
                solution = [val + deficit * (val / total) for val in solution]
        
        # Ensure the sum is exactly 1 after correction
        total = np.sum(solution)
        if not abs(total - 1.0) < 1e-5:
            raise ValueError("Probabilities are not validly normalized")
        # print(solution)

        solution_dict = dict(zip(self.availActAgent[position], solution))
        self.pi[position] = solution_dict
        return self.pi

    def updateBelief(self, action_a, position):
        """
        Updates the belief for the current state based on the opponent's action.
        Input:
            - action_a (str): Action performed by the opponent.
        Returns:
            - dict: Updated belief probabilities for the current state.
        """
        # print(self.history[position][action_a])
        self.history[position][action_a] = self.history[position][action_a] + 1
        self.counter[position] = self.counter[position] + 1
        
        for o in self.availActOpp[position]:
            self.B[position][o] = self.history[position][o]/self.counter[position]
        return self.B

    def updateBeliefValue(self, position):
        """
        Updates the belief value for the given position based on Q-values and belief probabilities.

        Args:
            position (str): Position string representing the state.

        Returns:
            dict: Updated belief value for the given position.
        """
        position = str(position)
        exp_values = []
        for a in self.availActAgent[position]:
            temp_exp_values = 0
            for o in self.availActOpp[position]:
                temp_exp_values += self.B[position][o] * self.Q_values[position][a][o] * self.pi[position][a]
            exp_values.append(temp_exp_values)
        
        self.V_values[position] = np.max(exp_values)
        return self.V_values

    def beliefBasedAction(self, action_a, action_b, position, new_position):
        self.B = self.updateBelief(action_a, position)

        # Update Q-value for the current action pair based on the reward and the V-value of the new position
        old_Q = self.Q_values[position][action_b][action_a]

        if self.opponent == False:
            self.Q_values[position][action_b][action_a] = (1 - self.lr) * old_Q + self.lr * (self.grid.reward + self.gamma * self.V_values[new_position])
        else:
            self.Q_values[position][action_b][action_a] = (1 - self.lr) * old_Q + self.lr * (-self.grid.reward + self.gamma * self.V_values[new_position])

        # Update policy based on the minimax algorithm
        self.pi = self.beliefBasedPolicy()

        # Update V-values for the current position
        old_V = self.V_values
        new_V = self.updateBeliefValue(position)
        self.V_values = new_V

        self.lr = self.lr * self.decay  

    def chooseBeliefAction(self):
        """
        Chooses an action using the belief-based policy with a probability of exploration.

        Returns:
            str: The chosen action.
        """
        # Variable to substitute with the chosen action
        action = ""

        # With probability exp_rate, completely randomic
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.availActAgent[str(self.grid.state)])

        # Otherwise choose an action with probability given by the policy
        else:
            temp_dict = self.pi[str(self.grid.state)]
            acts = list(temp_dict.keys())
            probs = list(temp_dict.values())
            action = np.random.choice(acts, p=probs)
        return action
    
############################################# 

    def trainMR(self, rounds = N_TRAIN):
        # self.explor = 0.2
        i = 0
        while i < rounds:
            positionB = str(self.grid.state)
            game_priority = random.randint(0, 1)

            actionB = self.chooseMinimaxAction()
            actionA = self.chooseRandomAction()

            newPositionB = str(self.grid.gameTurn(actionA, actionB, game_priority))

            self.minimaxAction(actionA, actionB, positionB, newPositionB)
            i = i + 1

        print("TRAIN-MR: agent won {} plays out of {}".format(self.grid.wonGames, self.grid.terminatedGames))
        print("victories: {} %".format(self.grid.wonGames/self.grid.terminatedGames*100))

    def testMR(self, rounds = N_TEST):
        i = 0        
        while i < rounds:
            game_priority = random.randint(0, 1)
            if self.opponent == False:
                action_b = self.chooseMinimaxAction()
                action_a = self.chooseRandomAction()
            else:
                action_a = self.chooseMinimaxAction()
                action_b = self.chooseRandomAction()

            self.grid.state = self.grid.gameTurn(action_a, action_b, game_priority)
            i = i + 1
        print("TEST-MR: agent won {} plays out of {}".format(self.grid.wonGames, self.grid.terminatedGames))
        print("victories: {} %".format(self.grid.wonGames/self.grid.terminatedGames*100))

#############################################

    def trainBR(self, rounds = N_TRAIN):
        i = 0
        while i < rounds:
            positionB = str(self.grid.state)
            game_priority = random.randint(0, 1)

            actionB = self.chooseBeliefAction()
            actionA = self.chooseRandomAction()

            newPositionB = str(self.grid.gameTurn(actionA, actionB, game_priority))

            self.beliefBasedAction(actionA, actionB, positionB, newPositionB)
            i = i + 1

        print("TRAIN-BR: agent won {} plays out of {}".format(self.grid.wonGames, self.grid.terminatedGames))
        print("victories: {} %".format(self.grid.wonGames/self.grid.terminatedGames*100))

    def testBR(self, rounds = N_TEST):
        i = 0
        while i < rounds:
            game_priority = random.randint(0, 1)
            
            if self.opponent == False:
                action_b = self.chooseBeliefAction()
                action_a = self.chooseRandomAction()
            else:
                action_a = self.chooseBeliefAction()
                action_b = self.chooseRandomAction()      

            self.grid.state = self.grid.gameTurn(action_a, action_b, game_priority)
            i = i + 1
            
        print("TEST-BR: agent won {} plays out of {}".format(self.grid.wonGames, self.grid.terminatedGames))
        print("victories: {} %".format(self.grid.wonGames/self.grid.terminatedGames*100))

