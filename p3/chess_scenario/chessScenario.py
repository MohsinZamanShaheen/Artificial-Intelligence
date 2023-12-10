#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue

import chess
import numpy as np
import sys
import heapq

from itertools import permutations


class chesssScenario():

    """
    A class to represent the game of chess.
    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW
        self.depthMax = 8
        self.checkMate = False

        # Prepare a dictionary to control the visited state and at which
        # depth they were found
        self.dictVisitedStates = {}
        # Dictionary to reconstruct the BFS path
        self.dictPath = {}


        # Nuevos parametros necesarios para aplicar Q-Learning 
        self.actions = 12
        self.num_filas = 8
        self.num_columnas = 8
        self.alpha = 0.1  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.1  # Exploración-Explotación trade-off
        self.Q = np.zeros((self.num_filas * self.num_columnas, self.actions)) # Tablero chess es 8x8 =64 por cada pieza(torre blanca y rey blanca)


    def getCurrentState(self):
        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
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

    def reconstructPath(self, state, depth):
        # When we found the solution, we obtain the path followed to get to this        
        for i in range(depth):
            self.pathToTarget.insert(0,state)
            #Per cada node, mirem quin és el seu pare
            state = self.dictPath[str(state)][0]

        self.pathToTarget.insert(0,state)

    def canviarEstat(self, start, to):
        # We check which piece has been moved from one state to the next
        if start[0] == to[0]:
            fitxaMogudaStart=1
            fitxaMogudaTo = 1
        elif start[0] == to[1]:
            fitxaMogudaStart = 1
            fitxaMogudaTo = 0
        elif start[1] == to[0]:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 1
        else:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 0
        # move the piece changed
        self.chess.moveSim(start[fitxaMogudaStart], to[fitxaMogudaTo])

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next for BFS we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.canviarEstat(moveList[i],moveList[i+1])


    def h(self,state):
        
        if state[0][2] == 2:
            posicioRei = state[1]
            posicioTorre = state[0]
        else:
            posicioRei = state[0]
            posicioTorre = state[1]
        # With the king we wish to reach configuration (2,4), calculate Manhattan distance
        fila = abs(posicioRei[0] - 2)
        columna = abs(posicioRei[1]-4)

        # Pick the minimum for the row and column, this is when the king has to move in diagonal
        # We calculate the difference between row an colum, to calculate the remaining movements
        # which it shoudl go going straight        
        hRei = min(fila, columna) + abs(fila-columna)
        # with the tower we have 3 different cases
        if posicioTorre[0] == 0 and (posicioTorre[1] < 3 or posicioTorre[1] > 5):
            hTorre = 0
        elif posicioTorre[0] != 0 and posicioTorre[1] >= 3 and posicioTorre[1] <= 5:
            hTorre = 2
        else:
            hTorre = 1
        # In our case, the heuristics is the real cost of movements
        return hRei + hTorre

    def AStarSearch(self, currentState):
        # The node root has no parent, thus we add None, and -1, which would be the depth of the 'parent node'
        # For this algorithm, I consider depth == g_value
        self.dictPath[str(currentState)] = (None, 0)
        frontera = []
        # Initally the g_value is 0
        gValueCurrentState = 0
        # Add the initial node to visited list
        self.listVisitedStates.append(currentState)
        # calculate initial node's f_value
        f_costCurrentState = gValueCurrentState + self.h(currentState)
        # push the initial state node into the frontera 
        heapq.heappush(frontera, (f_costCurrentState, currentState))
        
        # While the frontera contains elements, keep going
        while frontera:
            # extract from the frontera the state with minimum f_value. 
            f_cost_current, current_state = heapq.heappop(frontera)
            
            # switch piece position in states list
            if current_state[0][2] == 6:
                current_state = current_state[::-1]

            # get the g_value
            gValue_current_state = self.dictPath[str(current_state)][1]
            # base case to stop
            if gValue_current_state > self.depthMax:
                break
            # If it not the root node, we move the pieces from the previous to the current state
            if gValue_current_state > 0:
                self.movePieces(currentState, gValueCurrentState, current_state, gValue_current_state)

            # check if it is checkMate 
            if self.isCheckMate(current_state):
                self.reconstructPath(current_state, gValue_current_state)
                return True
            
            for successor in self.getListNextStatesW(current_state):
                # switch piece position in states list
                if successor[0][2] == 6:
                    successor = successor[::-1]   
                
                # the g_cost of successor will be the g_cost of current + 1
                suc_g_cost = gValue_current_state + 1
                f_successor = suc_g_cost + self.h(successor)     
                # if the successor is already visited
                if self.isVisited(successor):
                    # if the one visited previously is better or equal then this one, we skip this one and don't add it again
                    if self.dictPath[str(successor)][1] <= suc_g_cost:
                        continue
                    # if this one is better, add again
                    self.listVisitedStates.append(successor)
                    heapq.heappush(frontera, (f_successor, successor))
                # if not previously visited, add to frontera list and mark as visiteed
                else:  
                    self.listVisitedStates.append(successor)
                    heapq.heappush(frontera, (f_successor, successor))
                # update dictionary with parent and g_value to reconstruch path
                self.dictPath[str(successor)] = (current_state,suc_g_cost)
            # update states
            currentState = current_state
            gValueCurrentState = gValue_current_state
        return False

    def translate(s):
        """
        Translates traditional board coordinates of chess into list indices
        """

        try:
            row = int(s[0])
            col = s[1]
            if row < 1 or row > 8:
                print(s[0] + "is not in the range from 1 - 8")
                return None
            if col < 'a' or col > 'h':
                print(s[1] + "is not in the range from a - h")
                return None
            dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
            return (8 - row, dict[col])
        except:
            print(s + "is not in the format '[number][letter]'")
            return None
        
# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------- Q-LEARNING ------------------------------------------------ #
# ---------------------------------------------------------------------------------------------------- #

    def obtener_estado(self, fila, columna):
        return fila * self.num_columnas + columna
    
    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def es_accion_valida(self, state, action):

        tower_position = self.getPieceState(state, 2) # cojo posicion de torre a efectos de testing
        king_position = self.getPieceState(state, 6)

        new_state = []
        #KING
        if action == 0:  # Move up
            new_state = [(tower_position), [king_position[0] - 1, king_position[1], king_position[2]]]
        elif action == 1:  # Move down
            new_state = [(tower_position), [king_position[0] + 1, king_position[1], king_position[2]]]
        elif action == 2:  # Move left
            new_state = [(tower_position), [king_position[0], king_position[1] - 1, king_position[2]]]
        elif action == 3:  # Move right
            new_state = [(tower_position), [king_position[0], king_position[1] + 1, king_position[2]]]
        elif action == 4:  # Move diagonally up-left
            new_state = [(tower_position), [king_position[0] - 1, king_position[1] - 1, king_position[2]]]
        elif action == 5:  # Move diagonally up-right
            new_state = [(tower_position), [king_position[0] - 1, king_position[1] + 1, king_position[2]]]
        elif action == 6:  # Move diagonally down-left
            new_state = [(tower_position), [king_position[0] + 1, king_position[1] - 1, king_position[2]]]
        elif action == 7:  # Move diagonally down-right
            new_state = [(tower_position), [king_position[0] + 1, king_position[1] + 1, king_position[2]]]
        #TOWER
        elif action == 8: #Move UP
            new_state = [[tower_position[0]-1, tower_position[1], tower_position[2]], (king_position)]
        elif action == 9: #Move Down
            new_state = [[tower_position[0]+1, tower_position[1], tower_position[2]], (king_position)]
        elif action == 10: #Move Left
            new_state = [[tower_position[0], tower_position[1]-1, tower_position[2]], (king_position)]
        elif action == 11: #Move Right
            new_state = [[tower_position[0]-1, tower_position[1]+1, tower_position[2]], (king_position)]
        else:
            new_state = state

        new_tower_position = self.getPieceState(new_state, 2)
        new_king_position = self.getPieceState(new_state, 6)
        tower_position = self.getPieceState(state, 2)
        king_position = self.getPieceState(state, 6)

        if 8 <= action <= 11:
            if 0 <= new_tower_position[0] < 8 and 0 <= new_tower_position[1] < 8:
                correction_state = [(tower_position), (king_position)]
            else:
                correction_state = [(new_tower_position), (king_position)]
        elif 0 <= action <= 7:
            if 0 <= new_king_position[0] < 8 and 0 <= new_king_position[1] < 8:
                correction_state = [(tower_position), (king_position)]
            else:
                correction_state = [(tower_position), (new_king_position)]
        else:
            correction_state = state

        return correction_state

    def selection_action(self, stateActual):
        # Seleccionar una acción basada en epsilon-greedy
        if np.random.rand() < self.epsilon:
            accion = np.random.randint(self.actions)  # Exploración aleatoria
        else: 
            ## caso de explotación
            accion = np.argmax(self.Q[stateActual])
        return accion


    def isCheckMate(self, mystate):
        # list of possible check mate states
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False


    def reward(self, state):
        # si checkmante de blancas, recompensa 100
        if self.isCheckMate(state):
            return 100
        # en resto de posiciones -1
        else:
            return -1
        

    def q_learning_Chess(self, firstState):
        num_episodios = 1000
        umbral_convergencia = 0.01
        convergencias = 0
        num_convergencias_necesarias = 5

        for episodio in range(num_episodios):
            state_actual = firstState
            acciones = []

            # mientras no sea estado terminal
            while True:
                accion = self.selection_action(state_actual)

                # Realizar la acción y obtener la nueva posición
                new_state = self. es_accion_valida(state_actual, accion)

                recompensa = self.reward(new_state)

                self.Q[state_actual, accion] += self.alpha * (recompensa + self.gamma * np.max(self.Q[new_state]) - self.Q[state_actual, accion])

                # Actualizar la posición actual
                state_actual = new_state
                
                if self.isCheckMate(state_actual):
                    break
            

        # Imprimir la Q-table final
        print("\nQ-table final:")
        print(self.Q)

if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)

    # intiialize board
    TA = np.zeros((8, 8))
    # load initial state
    # white pieces
    TA[7][0] = 2
    TA[7][4] = 6
    # black pieces
    TA[0][4] = 12
    
    # initialise bord
    print("stating Chess Board... ")
    aichess = chesssScenario(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()
    print("printing board")
    aichess.chess.boardSim.print_board()


    print("Q-table:\n")
    print(aichess.Q)

    # get list of next states for current state
    print("current State",currentState,"\n")

    aichess.q_learning_Chess(currentState)

