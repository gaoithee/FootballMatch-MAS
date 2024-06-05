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

def trainBB(agentA, agentB, rounds = N_TRAIN):
    i = 0
    while i < rounds: 
        positionB = str(agentB.grid.state)

        game_priority = random.randint(0, 1)

        actionA = agentA.chooseBeliefAction()
        actionB = agentB.chooseBeliefAction()

        newPositionB = agentB.grid.gameTurn(actionA, actionB, game_priority)
        newPositionB = str(newPositionB)

        agentB.beliefBasedAction(actionA, actionB, positionB, newPositionB)
        agentA.beliefBasedAction(actionB, actionA, positionB, newPositionB)
        
        i = i + 1 

    print("TRAIN-BB: agentB won {} plays out of {}".format(agentB.grid.wonGames, agentB.grid.terminatedGames))
    print("victories: {} %".format(agentB.grid.wonGames/agentB.grid.terminatedGames*100))

###############################################

grid2 = State()
agentC = Agent(grid2, opponent_flag = True)
agentD = Agent(grid2)
chall1 = Challenger(grid2, opponent_flag=True)
chall2 = Challenger(grid2, opponent_flag=True)
chall3 = Challenger(grid2, opponent_flag=True)

print("TRAINING: belief vs belief")
trainBB(agentC, agentD, 2000000)

# Reset
agentD.grid.reward = 0
agentD.grid.wonGames = 0
agentD.grid.terminatedGames = 0
agentD.grid.state = [(3,1), (1,2)]

print("TESTING AGENT: BB vs random")
agentD.testBR()

###############################################

# Reset
agentD.grid.reward = 0
agentD.grid.wonGames = 0
agentD.grid.terminatedGames = 0
agentD.grid.state = [(3,1), (1,2)]

def trainCB(agent, challenger, rounds = 500000):
    i = 0
    game_priority = random.randint(0, 1)

    while i < rounds:        
        position = str(agent.grid.state)

        actionA = agent.chooseBeliefAction()
        actionC = challenger.chooseqLearningAction()

        newPosition = agent.grid.gameTurn(actionC, actionA, game_priority)
        newPosition = str(newPosition)
        
        challenger.qLearningAction(actionC, position, newPosition)
        
        i = i + 1 

    print("TRAIN-CB: challenger won {} plays out of {}".format(agent.grid.terminatedGames - agent.grid.wonGames, agent.grid.terminatedGames))
    print("victories: {} %".format((agent.grid.wonGames/agent.grid.terminatedGames)*100))

def testCB(agent, challenger, rounds = N_TEST):
    i = 0
    game_priority = random.randint(0, 1)
    
    while i < rounds:        

        actionA = agent.chooseBeliefAction()
        actionC = challenger.chooseqLearningAction()

        agent.grid.state = agent.grid.gameTurn(actionC, actionA, game_priority)

        i = i + 1 

    print("TEST-CM: challenger won {} plays out of {}".format(agent.grid.terminatedGames - agent.grid.wonGames, agent.grid.terminatedGames))
    print("victories: {} %".format((agent.grid.wonGames/agent.grid.terminatedGames)*100))

###############################################

print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

# Reset
agentD.grid.reward = 0
agentD.grid.wonGames = 0
agentD.grid.terminatedGames = 0
agentD.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 1 vs BB")
trainCB(agentD, chall1)

print("TESTING: challenger 1 vs BB")
testCB(agentD, chall1)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

# Reset
agentD.grid.reward = 0
agentD.grid.wonGames = 0
agentD.grid.terminatedGames = 0
agentD.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 2 vs BB")
trainCB(agentD, chall2)

print("TESTING: challenger 2 vs BB")
testCB(agentD, chall2)

#############################################
print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

# Reset
agentD.grid.reward = 0
agentD.grid.wonGames = 0
agentD.grid.terminatedGames = 0
agentD.grid.state = [(3,1), (1,2)]

print("TRAINING: challenger 3 vs BB")
trainCB(agentD, chall3)

print("TESTING: challenger 3 vs BB")
testCB(agentD, chall3)

#############################################



