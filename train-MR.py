# IMPORTS
import numpy as np
import pickle
import random as random
from grid import State
from agent_update import Agent 
from challenger import Challenger

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

#############################################

grid1 = State()
agent = Agent(grid1)
chall1 = Challenger(grid1, opponent_flag=True)
chall2 = Challenger(grid1, opponent_flag=True)
chall3 = Challenger(grid1, opponent_flag=True)

#############################################

print("5.000.000 - JUST TO TRY")
print("TRAINING: minimax vs random")
agent.trainMR(2000000)

agent.grid.reward = 0
agent.grid.wonGames = 0
agent.grid.terminatedGames = 0

print("TESTING AGENT: MR vs random")
agent.testMR()

#############################################

# Reset
agent.grid.reward = 0
agent.grid.wonGames = 0
agent.grid.terminatedGames = 0
agent.grid.state = [(3,1), (1,2)]

def trainCM(agent, challenger, rounds = 500000):
    i = 0
    game_priority = random.randint(0, 1)

    while i < rounds:        
        position = str(agent.grid.state)

        actionA = agent.chooseMinimaxAction()
        actionC = challenger.chooseqLearningAction()

        newPosition = agent.grid.gameTurn(actionC, actionA, game_priority)
        newPosition = str(newPosition)
        
        challenger.qLearningAction(actionC, position, newPosition)
        
        i = i + 1 

    print("TRAIN-CM: challenger won {} plays out of {}".format(agent.grid.terminatedGames - agent.grid.wonGames, agent.grid.terminatedGames))
    print("victories: {} %".format((agent.grid.wonGames/agent.grid.terminatedGames)*100))

def testCM(agent, challenger, rounds = 100000):
    i = 0
    game_priority = random.randint(0, 1)
    
    while i < rounds:        

        actionA = agent.chooseMinimaxAction()
        actionC = challenger.chooseqLearningAction()

        agent.grid.state = agent.grid.gameTurn(actionC, actionA, game_priority)

        i = i + 1 

    print("TEST-CM: challenger won {} plays out of {}".format(agent.grid.terminatedGames - agent.grid.wonGames, agent.grid.terminatedGames))
    print("victories: {} %".format((agent.grid.wonGames/agent.grid.terminatedGames)*100))

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

# Reset
agent.grid.reward = 0
agent.grid.wonGames = 0
agent.grid.terminatedGames = 0
agent.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 1 vs MR")
trainCM(agent, chall1)

print("TESTING: challenger 1 vs MR")
testCM(agent, chall1)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

agent.grid.reward = 0
agent.grid.wonGames = 0
agent.grid.terminatedGames = 0
agent.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 2 vs MR")
trainCM(agent, chall2)

print("TESTING: challenger 2 vs MR")
testCM(agent, chall2)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

agent.grid.reward = 0
agent.grid.wonGames = 0
agent.grid.terminatedGames = 0
agent.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 3 vs MR")
trainCM(agent, chall3)

print("TESTING: challenger 3 vs MR")
testCM(agent, chall3)

#############################################