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
N_TRAIN_dd = 250000
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

def trainMM(agentA, agentB, rounds = N_TRAIN):
    i = 0
    while i < rounds:
        positionB = agentB.grid.state
        positionB = str(positionB)
        # print(positionB)

        game_priority = random.randint(0, 1)
        actionA = agentA.chooseMinimaxAction()
        actionB = agentB.chooseMinimaxAction()

        newPositionB = agentB.grid.gameTurn(actionA, actionB, game_priority)
        newPositionB = str(newPositionB)

        agentA.minimaxAction(actionB, actionA, positionB, newPositionB)
        agentB.minimaxAction(actionA, actionB, positionB, newPositionB)
        
        i = i + 1 

    print("TRAIN-MM: agentB won {} plays out of {}".format(agentB.grid.wonGames, agentB.grid.terminatedGames))
    print("victories: {} %".format(agentB.grid.wonGames/agentB.grid.terminatedGames*100))

#############################################

grid1 = State()
agentA = Agent(grid1, opponent_flag = True)
agentB = Agent(grid1)
chall1 = Challenger(grid1, opponent_flag = True)
chall2 = Challenger(grid1, opponent_flag = True) 
chall3 = Challenger(grid1, opponent_flag = True) 

print("TRAINING: minimax vs minimax")
trainMM(agentA, agentB, 1000000)

agentB.grid.reward = 0
agentB.grid.wonGames = 0
agentB.grid.terminatedGames = 0

print("TESTING AGENT: MM vs random")
agentB.testMR()
print("TESTING AGENT: MM vs random")
agentA.testMRopp()

#############################################

# Reset
agentB.grid.reward = 0
agentB.grid.wonGames = 0
agentB.grid.terminatedGames = 0
agentB.grid.state = [(3,1), (1,2)]

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

print("TRAINING: challenger 1 vs MR")
trainCM(agentB, chall1)

print("TESTING: challenger 1 vs MR")
testCM(agentB, chall1)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

print("TRAINING: challenger 2 vs MR")
trainCM(agentB, chall2)

print("TESTING: challenger 2 vs MR")
testCM(agentB, chall2)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

print("TRAINING: challenger 3 vs MR")
trainCM(agentB, chall3)

print("TESTING: challenger 3 vs MR")
testCM(agentB, chall3)

#############################################