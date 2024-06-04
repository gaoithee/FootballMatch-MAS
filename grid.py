# IMPORTS
import numpy as np
import random as random

# GLOBAL VARIABLES

# how the grid is made:
BOARD_ROWS = 4
BOARD_COLS = 5
# in which states there could happen a win or a lose
POTENTIAL_LOSE_STATE_1 = (0, 1)
POTENTIAL_LOSE_STATE_2 = (0, 2)
POTENTIAL_WIN_STATE_1 = (4, 2)
POTENTIAL_WIN_STATE_2 = (4, 1) 
# initial positions of the two players
START_A = (3, 1)
START_B = (1, 2)


class State:
    
    # init method definining the grid and the initial positions of the two players
    def __init__(self, state=[START_A, START_B]):

        # initialize a matrix of zeros
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        # except from the positions of the two players!
        self.board[state[0][0]][state[0][1]] = 3 # the opponent is initialized to 3
        self.board[state[1][0]][state[1][1]] = 2 # the player to 2
        
        # we need to keep track of both the states of the agents
        # note: `state[0]` represents the position (x, y) of the opponent, `state[1]` of the player
        self.state = [state[0], state[1]] # `self.state` is the couple of the agents' coordinates

        # and also we need the information of who has the ball!
        self.b_has_ball = True 

        # boolean variables to know when to stop the game
        self.isEnd = False

        # reward
        self.reward = 0

        # counters of: terminated and won games
        self.terminatedGames = 0
        self.wonGames = 0

        # prints the game board if required
    
    # function to print the game board (not used)
    def showBoard(self):
        """
        Simply prints the board if called.
        This is the output if used on a new grid, where the capital letter denotes the ball possession:
        ---------------------
        | 0 | 0 | 0 | 0 | 0 |
        | 0 | 0 | 0 | a | 0 |
        | 0 | B | 0 | 0 | 0 |
        | 0 | 0 | 0 | 0 | 0 |
        ---------------------

        Board Layout:
            ----------------------------------------------
            | (0, 0) | (1, 0) | (2, 0) | (3, 0) | (4, 0) |
            | (0, 1) | (1, 1) | (2, 1) | (3, 1) | (4, 1) |
            | (0, 2) | (1, 2) | (2, 2) | (3, 2) | (4, 2) |
            | (0, 3) | (1, 3) | (2, 3) | (3, 3) | (4, 3) | 
            ----------------------------------------------
        """
        # BOARD_ROWS = 4, BOARD_COLS = 5
        for i in range(0, BOARD_ROWS):
            print('---------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 0:
                    token = '0'
                if j == self.state[1][0] and i == self.state[1][1]:
                    token = 'b'
                    if self.b_has_ball == True:
                        token = 'B'
                if j == self.state[0][0] and i == self.state[0][1]:
                    token = 'a'
                    if self.b_has_ball == False:
                        token = 'A'
                out += token + ' | '
            print(out)
        print('---------------------')
    
    # returns whether or not the game should end, the reward, the number of terminated, won and lost plays
    def gameShouldEnd(self, action_a, action_b):
        """
        Determines if the game should end based on the actions performed by the opponent and the player.
        
        Input:
            - `action_a`: action performed by the opponent
            - `action_b`: action performed by the player
        
        Returns:
            A tuple containing:
            - `self.isEnd`: boolean indicating if the game should end
            - `self.reward`: the current reward
            - `self.terminatedGames`: the number of terminated games
            - `self.wonGames`: the number of games won by the player
        """
        # Initialize win conditions to False
        a_should_win = False
        b_should_win = False

        # Evaluate if the opponent (A) should win
        a_should_win = (
            (self.state[0] == POTENTIAL_LOSE_STATE_1 or self.state[0] == POTENTIAL_LOSE_STATE_2) 
            and action_a == 'W' 
            and not self.b_has_ball
        )

        # Evaluate if the player (B) should win
        b_should_win = (
            (self.state[1] == POTENTIAL_WIN_STATE_1 or self.state[1] == POTENTIAL_WIN_STATE_2) 
            and action_b == 'E' 
            and self.b_has_ball
        )

        # If the player should win
        if b_should_win:
            self.isEnd = True
            self.reward += 1
            self.terminatedGames += 1
            self.wonGames += 1

        # If the opponent should win
        elif a_should_win:
            self.isEnd = True
            self.reward -= 1  # since the player lost the game
            self.terminatedGames += 1

        return self.isEnd, self.reward, self.terminatedGames, self.wonGames

    # proposes the next position of the agent on which is invoked
    # note: "proposes" because we need to perform more checks before effectively moving the agent!
    def nextPosition(self, state, action):
        """
        Determines the candidate next position of the agent based on a given action.
        
        Input:
            - `state`: the current coordinates of the agent (x, y)
            - `action`: the action to be taken, which can be "N" (north), "S" (south), "E" (east), "W" (west), or "stand" (no movement)
        
        Board Layout:
            ----------------------------------------------
            | (0, 0) | (1, 0) | (2, 0) | (3, 0) | (4, 0) |
            | (0, 1) | (1, 1) | (2, 1) | (3, 1) | (4, 1) |
            | (0, 2) | (1, 2) | (2, 2) | (3, 2) | (4, 2) |
            | (0, 3) | (1, 3) | (2, 3) | (3, 3) | (4, 3) | 
            ----------------------------------------------
        
        Returns:
            The next position of the agent after performing the action.
        """
        # Determine the next position based on the action
        if action == "N":
            nextState = (state[0], state[1] - 1)
        elif action == "S":
            nextState = (state[0], state[1] + 1)
        elif action == "W":
            nextState = (state[0] - 1, state[1])
        elif action == "E":
            nextState = (state[0] + 1, state[1])
        else:
            nextState = state  # "stand" action or any other invalid action results in no movement
        
        # Ensure the next position is within the board limits
        if 0 <= nextState[0] <= BOARD_COLS - 1 and 0 <= nextState[1] <= BOARD_ROWS - 1:
            # self.board[state[0]][state[1]] = 0
            state = nextState  # Remove the player from the current position on the board

        return state

    # FOUL = when a player executes an action that would take it to the square OCCUPIED (i.e. already occupied!) by the other player, 
    # possession of the ball goes to the STATIONARY player and THE MOVE DOES NOT TAKE PLACE
    def checkFoul(self, action_a, action_b, game_priority):
        # IMPORTANT: here we assume that even if both the player and the opponent commit a foul
        # (i.e. B wants to go above A, but also A wants to go above B) then only the first one playing 
        # receives a foul. I do not see whether one of the two stood still or not!
        """
        Checks if a foul occurs when either player attempts to move to a position occupied by the other player.
        
        Input:
            - `action_a`: action performed by the opponent
            - `action_b`: action performed by the player
            - `game_priority`: determines which player has priority (relevant only if both commit a foul)
        
        Returns:
            A tuple containing:
            - `foulOnA`: boolean indicating if a foul was committed on player A
            - `foulOnB`: boolean indicating if a foul was committed on player B
            - `self.b_has_ball`: boolean indicating if player B has the ball
        """

        # Calculate the candidate positions for both players
        candidateNextPosA = self.nextPosition(self.state[0], action_a)
        candidateNextPosB = self.nextPosition(self.state[1], action_b)

        # Initialize foul indicators
        foulOnA = False
        foulOnB = False

        # Check if both players are trying to move to each other's current position
        if candidateNextPosB == self.state[0] and candidateNextPosA == self.state[1]:
            foulOnA = True
            foulOnB = True
            if game_priority == 1:
                self.b_has_ball = False
            else:
                self.b_has_ball = True

        # Check if player A is trying to move to player B's position since it moves first
        elif candidateNextPosA == self.state[1] and game_priority == 0:
            foulOnB = True
            self.b_has_ball = True

        # Check if player B is trying to move to player A's position since it moves first
        elif candidateNextPosB == self.state[0] and game_priority == 1:
            foulOnA = True
            self.b_has_ball = False

	# Check if player A is trying to move to player B's new position since it moves second
        elif candidateNextPosA == candidateNextPosB and game_priority == 1:
            foulOnB = True
            self.b_has_ball = True

        # Check if player B is trying to move to player A's new position since it moves second
        elif candidateNextPosB == candidateNextPosA and game_priority == 0:
            foulOnA = True
            self.b_has_ball = False


        return foulOnA, foulOnB, self.b_has_ball

    # performs (concretely) the game turn, given the two agent's actions 
    def gameTurn(self, action_a, action_b, game_priority):
        """
        Executes a game turn based on the actions of the opponent and the player.
        
        Input:
            - `action_a`: action performed by the opponent
            - `action_b`: action performed by the player
            - `game_priority`: who moves first
        
        Returns:
            The updated state of the game.
        """
        
        if self.isEnd:
            # Reset the initial configuration if the game has ended
            self.state[0] = START_A
            self.state[1] = START_B
            # But give the ball possession randomly 
            self.b_has_ball = random.choice([True, False])
            self.isEnd = False
    
        else:
            # Save the original positions
            original_a = self.state[0]
            original_b = self.state[1]

            # Check for fouls and eventually change the ball possession
            foulOnA, foulOnB, self.b_has_ball = self.checkFoul(action_a, action_b, game_priority)

            # No player commits a foul
            if not foulOnA and not foulOnB:
                if game_priority:
                    # Player B has the game priority and no foul was committed
                    self.state[1] = self.nextPosition(self.state[1], action_b)
                    self.state[0] = self.nextPosition(self.state[0], action_a)
                else:
                    # Player A has the game priority and no foul was committed
                    self.state[0] = self.nextPosition(self.state[0], action_a)                
                    self.state[1] = self.nextPosition(self.state[1], action_b)
            
            # Both players committed a foul, so no one moves
            elif foulOnA and foulOnB:
                self.state[0] = original_a
                self.state[1] = original_b
            
            # Player B committed a foul on Player A, so Player A moves and Player B stays
            elif foulOnA:
                self.state[0] = self.nextPosition(self.state[0], action_a)
                self.state[1] = original_b

            # Player A committed a foul on Player B, so Player B moves and Player A stays
            else:
                self.state[1] = self.nextPosition(self.state[1], action_b)
                self.state[0] = original_a

        # Check if the game should end and update the relevant attributes
        self.isEnd, self.reward, self.terminatedGames, self.wonGames = self.gameShouldEnd(action_a, action_b)

        return self.state

