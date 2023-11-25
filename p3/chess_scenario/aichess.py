#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
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


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

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
        self.checkMate = False

    def copyState(self, state):

        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def isVisitedSituation(self, color, mystate):

        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False

    def getCurrentStateW(self):

        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si les negres maten el rei blanc, no és una configuració correcta
        if wkState == None:
            return False
        # Mirem les possibles posicions del rei blanc i mirem si en alguna pot "matar" al rei negre
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Tindríem un checkMate
                return True
        if wrState != None:
            # Mirem les possibles posicions de la torre blanca i mirem si en alguna pot "matar" al rei negre
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]

        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If whites kill the black king , it is not a correct configuration
        if bkState == None:
            return False
        # We check all possible positions for the black king, and chck if in any of them it may kill the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # That would be checkMate
                return True
        if brState != None:
            # We check the possible positions of the black tower, and we chck if in any of them it may kill the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

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

    def heuristica(self, currentState, color):
        # In this method, we calculate the heuristics for both the whites and black ones
        # The value calculated here is for the whites,
        # but finally from verything, as a function of the color parameter, we multiply the result by -1
        value = 0

        bkState = self.getPieceState(currentState, 12)
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)
        brState = self.getPieceState(currentState, 8)

        # Getting row and columns for each piece
        filaBk = bkState[0]
        columnaBk = bkState[1]
        filaWk = wkState[0]
        columnaWk = wkState[1]

        if wrState != None:
            filaWr = wrState[0]
            columnaWr = wrState[1]
        if brState != None:
            filaBr = brState[0]
            columnaBr = brState[1]

        # We check if they killed the black tower
        if brState == None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)
            if distReis >= 3 and wrState != None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the closer our king from the oponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # If not, we will only prioritize that he approahces the wall, to be able to approach the check mate
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # They killed the black tower.
        # Within this method we consider the same conditions than in the previous section, but now with reversed values.
        if wrState == None:
            value += -50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState != None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the close we have our king from the oponent, the better
            # If we substract 7 to the distance between both kings, as this is the max distance they can be at in a chess board
            value += (-7 + distReis)

            if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # We are checking blacks
        if self.isWatchedBk(currentState):
            value += 20

        # We are checking whites
        if self.isWatchedWk(currentState):
            value += -20

            # If black, values are negative, otherwise positive
        if not color:
            value = (-1) * value

        return value

    # Helper method that checks if a black rook can be eliminated
    def eliminarBlack(self, blackState, brState, successor):
        self.newBoardSim(blackState)
        newBlackState = blackState.copy()
        if brState != None:
            if len(successor) >= 2:
                if brState[0:2] == successor[0][0:2] or brState[0:2] == successor[1][0:2]:
                    newBlackState.remove(brState)
            else:
                if brState[0:2] == successor[0][0:2]:
                    newBlackState.remove(brState)
        return newBlackState

    # Helper method that checks if a white rook can be eliminated
    def eliminarWhite(self, whiteState, wrState, successor):
        self.newBoardSim(whiteState)
        newWhiteState = whiteState.copy()
        if wrState != None:
            if len(successor) >= 2:
                if wrState[0:2] == successor[0][0:2] or wrState[0:2] == successor[1][0:2]:
                    newWhiteState.remove(wrState)
            else:
                if wrState[0:2] == successor[0][0:2]:
                    newWhiteState.remove(wrState)
        return newWhiteState

    # Method to check checkMate cases to stop the algorithm
    def isCheckMate(self, state):
        self.newBoardSim(state)
        brState = self.getPieceState(state, 8)
        wrState = self.getPieceState(state, 2)
        whiteState = self.getWhiteState(state)
        blackState = self.getBlackState(state)

        for successor in self.getListNextStatesW(whiteState):
            successor += self.eliminarBlack(blackState, brState, successor)
            if not self.isWatchedWk(successor):
                self.newBoardSim(state)
                return False

        for successor in self.getListNextStatesB(blackState):
            successor += self.eliminarWhite(whiteState, wrState, successor)
            if not self.isWatchedBk(successor):
                self.newBoardSim(state)
                return False

        return True

# ---------------------- MINIMAX START  --------------------------- #

    def minimaxGame(self, depthWhite, depthBlack, playerTurn):
        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(currentState):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)

            # check player turn
            if playerTurn:
                movimiento = self.minimax(currentState, depthWhite, depthWhite, playerTurn)
            else:
                movimiento = self.minimax(currentState, depthBlack, depthBlack, playerTurn)

            if (movimiento is None):
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            #in case the pieces are repeting movements, stop 
            if (self.isVisitedSituation(playerTurn, self.copyState(movimiento))):
                return print("JUEGO EN TABLAS")

            self.listVisitedSituations.append((playerTurn, self.copyState(movimiento)))

            # make best movement and print on board
            piece_moved = self.getMovement(currentState, self.copyState(movimiento))
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn


    def minimax(self, state, depth, depthColor, playerTurn):

        # check if it is ternimal node or checkmate scenario to return static heuristic value
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)
        
        # variable that will contain the best movement to make
        maxState = None

        # Maximizing player
        if playerTurn:
            currBestValue = float('-inf')

            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            # We see the successors only for the states in White
            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)

                if not self.isWatchedWk(successor):
                    #self.newBoardSim(state)
                    bestValue = self.minimax(successor, depth - 1, depthColor, False)
                    # check for best value and best movement if any
                    if bestValue > currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # Minimizing player
        else:
            # initialize minimizer
            currBestValue = float('inf')
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            # We see the successors only for the states in Black
            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)

                if not self.isWatchedBk(successor):
                    #self.newBoardSim(state)
                    # Recursively call minimax with the successor state
                    bestValue = self.minimax(successor, depth - 1, depthColor, True)

                    # Update the best value and maxState if a better successor is found
                    if bestValue < currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # if back to top level, return the best movement
        if depth == depthColor:
            return maxState

        return currBestValue
    
# ---------------------- MINIMAX END  --------------------------- #




# --------------------- PODA ALPHA BETA START ---------------------- #

    def alphaBetaPoda(self, depthWhite, depthBlack, playerTurn, alpha=-float('inf'), beta=float('inf')):

        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(self.copyState(currentState)):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)
            if playerTurn:
                movimiento = self.minimaxalfabeta(currentState, depthWhite, depthWhite, playerTurn, alpha, beta)

            else:
                movimiento = self.minimaxalfabeta(currentState, depthBlack, depthBlack, playerTurn, alpha, beta)

            if (movimiento is None):
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            if (self.isVisitedSituation(playerTurn, self.copyState(movimiento))):
                return print("JUEGO EN TABLAS")
            
            self.listVisitedSituations.append((playerTurn, self.copyState(movimiento)))

            # make best movement and print on board
            piece_moved = self.getMovement(currentState, self.copyState(movimiento))
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn


    def minimaxalfabeta(self, state, depth, depthColor, playerTurn, alpha, beta):
         # check if it is ternimal node or checkmate scenario to return static heuristic value
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)
        
        # variable that will contain the best movement to make
        maxState = None

        # Maximizing player
        if playerTurn:
            currBestValue = float('-inf')

            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            # We see the successors only for the states in White
            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)

                if not self.isWatchedWk(successor):
                    #self.newBoardSim(state)
                    # Recursively call with the successor state
                    bestValue = self.minimaxalfabeta(successor, depth - 1, depthColor, False, alpha, beta)

                    # Update the best value and maxState if a better successor is found
                    if bestValue > currBestValue:
                        currBestValue = bestValue
                        maxState = successor

                    # Update alpha (the best value for the maximizing player)
                    alpha = max(alpha, currBestValue)

                    # Check if we can prune the search
                    if beta <= alpha:
                        break

        # Minimizing player
        else:
            currBestValue = float('inf')
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            # We see the successors only for the states in Black
            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)

                if not self.isWatchedBk(successor):
                    #self.newBoardSim(state)
                    # Recursively call with the successor state
                    bestValue = self.minimaxalfabeta(successor, depth - 1, depthColor, True, alpha, beta)

                    # Update the best value and maxState if a better successor is found
                    if bestValue < currBestValue:
                        currBestValue = bestValue
                        maxState = successor
                    # Update beta which is best for minimizing player
                    beta = min(beta, currBestValue)

                    if beta <= alpha:
                        break
        '''
        if maxState == None:
            if self.isCheckMate(state):
                return self.heuristica(state, playerTurn)
        '''
        # if back to top level, return the best movement
        if depth == depthColor:
            return maxState

        return currBestValue

# ---------------------- PODA ALPHA BETA END ---------------------- #


# ---------------------- EXPECTIMAX START  --------------------------- #

    def expectimax(self, depthWhite, depthBlack, playerTurn):
        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(self.copyState(currentState)):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)
            if playerTurn:
                movimiento = self.minimaxExpect(currentState, depthWhite, depthWhite, playerTurn)

            else:
                movimiento = self.minimaxExpect(currentState, depthBlack, depthBlack, playerTurn)
            
            print("MOVIMIENTO: ", movimiento)
            if (movimiento is None):# quan ja no hi ha mes moviments a fer
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            if (self.isVisitedSituation(playerTurn, movimiento)):
                return print("JUEGO EN TABLAS")
            
            self.listVisitedSituations.append((playerTurn, movimiento))
            # make best movement and print on board
            piece_moved = self.getMovement(currentState, movimiento)
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn


    def minimaxExpect(self, state, depth, depthColor, playerTurn):
        # check if it is ternimal node or checkmate scenario to return static heuristic value
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)
        
        maxState = None

        if playerTurn:
            currBestValue = float('-inf')

            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            # We see the successors only for the states in White
            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)

                if not self.isWatchedWk(successor):
                    #self.newBoardSim(state)
                    bestValue = self.minimaxExpect(successor, depth - 1, depthColor, False)
                    if bestValue >= currBestValue:
                        currBestValue = bestValue
                        maxState = successor
        
        else:
            possibleValues = []
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            # We see the successors only for the states in Black
            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)

                if not self.isWatchedBk(successor):
                    #self.newBoardSim(state)
                    bestValue = self.minimaxExpect(successor, depth - 1, depthColor, True)

                    # add value to the list
                    if bestValue is not None:
                        possibleValues.append(bestValue)
                        maxState = successor
            # if list have values, calculate the expected value
            if len(possibleValues) > 0:
                bestValue = self.calculateValue(possibleValues)
            
            # otherwise return a minimum value
            else:
                bestValue = float('-inf')

        if depth == depthColor:
            return maxState
            
        return bestValue

# ---------------------- EXPECTIMAX END  --------------------------- #

    def mitjana(self, values):
        sum = 0
        N = len(values)
        for i in range(N):
            sum += values[i]

        return sum / N

    def desviacio(self, values, mitjana):
        sum = 0
        N = len(values)

        for i in range(N):
            sum += pow(values[i] - mitjana, 2)

        return pow(sum / N, 1 / 2)

    def calculateValue(self, values):

        if len(values) == 0:
            return 0
        mitjana = self.mitjana(values)
        desviacio = self.desviacio(values, mitjana)
        # If deviation is 0, we cannot standardize values, since they are all equal, thus probability willbe equiprobable
        if desviacio == 0:
            # We return another value
            return values[0]

        esperanca = 0
        sum = 0
        N = len(values)
        for i in range(N):
            # Normalize value, with mean and deviation - zcore
            normalizedValues = (values[i] - mitjana) / desviacio
            # make the values positive with function e^(-x), in which x is the standardized value
            positiveValue = pow(1 / math.e, normalizedValues)
            # Here we calculate the expected value, which in the end will be expected value/sum            
            # Our positiveValue/sum represent the probabilities for each value
            # The larger this value, the more likely
            esperanca += positiveValue * values[i]
            sum += positiveValue

        return esperanca / sum
    

    def minimaxalphabetaGame(self, depthWhite, depthBlack, playerTurn, alpha=-float('inf'), beta=float('inf')):
        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(currentState):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)

            if playerTurn:
                movimiento = self.minimax(currentState, depthWhite, depthWhite, playerTurn)
            else:
                movimiento = self.minimaxalfabeta(currentState, depthBlack, depthBlack, playerTurn, alpha, beta)

            if (movimiento is None):
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            if (self.isVisitedSituation(playerTurn, self.copyState(movimiento))):
                return print("JUEGO EN TABLAS")

            self.listVisitedSituations.append((playerTurn, self.copyState(movimiento)))

            piece_moved = self.getMovement(currentState, self.copyState(movimiento))
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn


    def expectimaxAlphabetaGame(self, depthWhite, depthBlack, playerTurn, alpha=-float('inf'), beta=float('inf')):
        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(currentState):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)

            if playerTurn:
                movimiento = self.minimaxExpect(currentState, depthWhite, depthWhite, playerTurn)
            else:
                movimiento = self.minimaxalfabeta(currentState, depthBlack, depthBlack, playerTurn, alpha, beta)

            if (movimiento is None):
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            if (self.isVisitedSituation(playerTurn, self.copyState(movimiento))):
                return print("JUEGO EN TABLAS")

            self.listVisitedSituations.append((playerTurn, self.copyState(movimiento)))

            piece_moved = self.getMovement(currentState, self.copyState(movimiento))
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn



if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))

    # Configuració inicial del taulell
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][7] = 8
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    print("blackstate")
    print(aichess.currentStateB)

    print("whitestate")
    print(aichess.currentStateW)
    print("printing board")
    aichess.chess.boardSim.print_board()

    # Exercises

    # Run exercise 1
    playerTurn = True  # Whites start first = True

    # Exercise 1:To execute minimax for both players, uncomment the next line
    aichess.minimaxGame(4, 4, playerTurn)

    #To execute alpha beta pruning on both whites and blacks, uncomment the next line 
    #aichess.alphaBetaPoda(4,4, playerTurn)

    #To execute expectimax on both whites and blacks, uncomment the next line
    #aichess.expectimax(4,4,playerTurn)

    #Exercise 3: To execute Alpha beta on blacks and minimax on whites, uncomment the next line.
    #aichess.minimaxalphabetaGame(4, 4, playerTurn)

    #Exercise 5: To execute expectimax on whites and alpha-beta on blacks, uncomment the next line
    #aichess.expectimaxAlphabetaGame(3, 3, playerTurn)
