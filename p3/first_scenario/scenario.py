import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List
RawStateType = List[List[List[int]]]

from itertools import permutations

class Scenario():
    def __init__(self, TA, myinit=True):
        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW
        self.currentStateB = self.chess.boardSim.currentStateB
        self.goal = False
        self.goalState = None
        self.goal_state = None
        self.reward = 0

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def copyState(self, state):

        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState



    def q_learning(self, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Algoritmo Q-learning
        current_state = self.getCurrentState()
        pass


if __name__ == "__main__":
    # Initialize the scenario
    TA = np.zeros((3, 4))
    TA[2][0] = 12


    scenario = Scenario(TA, True)
    scenario.goalState = [0, 3, 12]

    scenario.chess.boardSim.print_board()

    # Run Q-learning algorithm
    q_table = scenario.q_learning(episodes=1000)

    # Print the first, two intermediate, and final Q-table
    print("First Q-table:")
    print(q_table)

    # Run additional episodes and print intermediate Q-tables
    for _ in range(2):
        q_table = scenario.q_learning(episodes=500)
        print("Intermediate Q-table:")
        print(q_table)

    # Run the final episodes and print the final Q-table
    q_table = scenario.q_learning(episodes=1000)
    print("Final Q-table:")
    print(q_table)