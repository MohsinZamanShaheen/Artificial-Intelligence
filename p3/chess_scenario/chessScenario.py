#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue

import chess
import numpy as np
import sys
import heapq
import random

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
        self.actions = 8 + 28
        self.num_filas = 8
        self.num_columnas = 8
        self.alpha = 0.4  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.4  # Exploración-Explotación trade-off
        #self.Q = np.zeros((self.num_filas * self.num_columnas, self.actions)) # Tablero chess es 8x8 =64 por cada pieza(torre blanca y rey blanca)
        self.Q= {}
        for row in range(self.num_filas):
            for col in range(self.num_columnas):
                state_key = tuple(map(tuple, self.currentStateW))
                self.Q[state_key] = {}
                for action in range(self.actions):
                    self.Q[state_key][action] = 0

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


    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        # load initial state
        # white pieces
        TA[7][0] = 2
        TA[7][4] = 6
        # black pieces
        TA[0][4] = 12

        self.chess.newBoardSim(TA)
    
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
        black_king = [[0,4,12]]
        new_state = []

        if 0 <= action <=7:
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
        else:
            #TOWER
            #UP MOVEMENTS
            if action == 8: #Move UP 1
                new_state = [[tower_position[0]-1, tower_position[1], tower_position[2]], (king_position)]
            elif action == 9: #Move UP 2
                new_state = [[tower_position[0]-2, tower_position[1], tower_position[2]], (king_position)]
            elif action == 10: #Move UP 3
                new_state = [[tower_position[0]-3, tower_position[1], tower_position[2]], (king_position)]
            elif action == 11: #Move UP 4
                new_state = [[tower_position[0]-4, tower_position[1], tower_position[2]], (king_position)]
            elif action == 12: #Move UP 5
                new_state = [[tower_position[0]-5, tower_position[1], tower_position[2]], (king_position)]
            elif action == 13: #Move UP 6
                new_state = [[tower_position[0]-6, tower_position[1], tower_position[2]], (king_position)]
            elif action == 14: #Move UP 7
                new_state = [[tower_position[0]-7, tower_position[1], tower_position[2]], (king_position)]

            # DOWN MOVEMENTS
            elif action == 15: #Move Down 1
                new_state = [[tower_position[0]+1, tower_position[1], tower_position[2]], (king_position)]
            elif action == 16: #Move Down 2
                new_state = [[tower_position[0]+2, tower_position[1], tower_position[2]], (king_position)]
            elif action == 17: #Move Down 3
                new_state = [[tower_position[0]+3, tower_position[1], tower_position[2]], (king_position)]
            elif action == 18: #Move Down 4
                new_state = [[tower_position[0]+4, tower_position[1], tower_position[2]], (king_position)]
            elif action == 19: #Move Down 5
                new_state = [[tower_position[0]+5, tower_position[1], tower_position[2]], (king_position)]
            elif action == 20: #Move Down 6
                new_state = [[tower_position[0]+6, tower_position[1], tower_position[2]], (king_position)]
            elif action == 21: #Move Down 7
                new_state = [[tower_position[0]+7, tower_position[1], tower_position[2]], (king_position)]

            # LEFT MOVEMENTS
            elif action == 22: #Move Left 1
                new_state = [[tower_position[0], tower_position[1]-1, tower_position[2]], (king_position)]
            elif action == 23: #Move Left 2
                new_state = [[tower_position[0], tower_position[1]-2, tower_position[2]], (king_position)]
            elif action == 24: #Move Left 3
                new_state = [[tower_position[0], tower_position[1]-3, tower_position[2]], (king_position)]
            elif action == 25: #Move Left 4
                new_state = [[tower_position[0], tower_position[1]-4, tower_position[2]], (king_position)]
            elif action == 26: #Move Left 5
                new_state = [[tower_position[0], tower_position[1]-5, tower_position[2]], (king_position)]
            elif action == 27: #Move Left 6
                new_state = [[tower_position[0], tower_position[1]-6, tower_position[2]], (king_position)]
            elif action == 28: #Move Left 7
                new_state = [[tower_position[0], tower_position[1]-7, tower_position[2]], (king_position)]

            # RIGHT MOVEMENTS
            elif action == 29: #Move Right 1
                new_state = [[tower_position[0], tower_position[1]+1, tower_position[2]], (king_position)]
            elif action == 30: #Move Right 2
                new_state = [[tower_position[0], tower_position[1]+2, tower_position[2]], (king_position)]
            elif action == 31: #Move Right 3
                new_state = [[tower_position[0], tower_position[1]+3, tower_position[2]], (king_position)]
            elif action == 32: #Move Right 4
                new_state = [[tower_position[0], tower_position[1]+4, tower_position[2]], (king_position)]
            elif action == 33: #Move Right 5
                new_state = [[tower_position[0], tower_position[1]+5, tower_position[2]], (king_position)]
            elif action == 34: #Move Right 6
                new_state = [[tower_position[0], tower_position[1]+6, tower_position[2]], (king_position)]
            elif action == 35: #Move Right 7
                new_state = [[tower_position[0], tower_position[1]+7, tower_position[2]], (king_position)]
            else:
                new_state = state

        new_tower_position = self.getPieceState(new_state, 2)
        new_king_position = self.getPieceState(new_state, 6)
        '''
        if 8 <= action <= 35:
            if 0 <= new_tower_position[0] <= 7 and 0 <= new_tower_position[1] <= 7:
                correction_state = [(new_tower_position), (king_position)]
            else:
                correction_state = [(tower_position), (king_position)]
        elif 0 <= action <= 7:
            if 0 <= new_king_position[0] <= 7 and 0 <= new_king_position[1] <= 7:
                correction_state = [(tower_position), (new_king_position)]
            else:
                correction_state = [(tower_position), (king_position)]
        else:
            correction_state = state
        '''
        if new_state[0][0] == black_king[0][0] and new_state[0][1] == black_king[0][1]:
            return [(tower_position), (king_position)]
        #print("De estado: ", state, " con action ", action, " al estado ", correction_state )

        return new_state

    def print_best_path(self, start_state):
        self.newBoardSim(start_state)
        current_state = start_state
        path = [current_state]
        path_found = True

        while not self.isCheckMate(current_state):
            state_key = self.state_to_tuple(current_state)
            #print(self.q_table[state_key])
            
            if state_key not in self.Q:
                break
            
            # Get the action with the highest Q-value for the current state
            best_action = max(self.Q[state_key], key=self.Q[state_key].get)
            next_state = self.es_accion_valida(current_state,best_action)
            if current_state != next_state:  
                movement = self.getMovement(current_state, next_state)
                self.chess.moveSim(movement[0], movement[1])
                #self.chess.boardSim.print_board()
                
            else:
                path_found = False
                print("No path found! Retry with different parametres. Agent needs more learning.")
                break


            current_state = next_state
            path.append(current_state)

        # If we have found the path, we print it
        if path_found:
            # Print the final path
            print("\nBest path to checkmate:")
            print(path)
    
    
    def kingPosition(self, state):
        king_pos = self.getPieceState(state, 6)

        return king_pos[0], king_pos[1]

    def towerPosition(self, state):
        tower_pos = self.getPieceState(state, 2)

        return tower_pos[0], tower_pos[1]
    
    def state_to_tuple(self, state):
        return tuple(map(tuple, state))
    
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

    def selection_action(self, stateActual):
        # Seleccionar una acción basada en epsilon-greedy

        if np.random.rand() < self.epsilon:
            # Exploración aleatoria
            accion = np.random.randint(self.actions)
        else:
            # Explote
            state = self.state_to_tuple(stateActual)
            q_values = self.Q[state]
            accion = max(q_values, key=q_values.get)
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
            return -1 *self.h(state)

    def q_learning_Chess(self, firstState):
        num_episodios = 1000
        umbral_convergencia = 0.0001
        convergencias = 0
        num_convergencias_necesarias = 5

        for episodio in range(num_episodios):
            print("episodi: ", episodio)
            currentState = firstState
            self.newBoardSim(currentState)
            acciones = []
            end = False

            while not self.isCheckMate(currentState):
                
                action = self.selection_action(currentState)
                nextPossibleStates = self.getListNextStatesW(currentState)
                nextState = self.es_accion_valida(currentState,action)
                same_next_State = nextState[::-1] 
                if ((nextState not in nextPossibleStates) and (same_next_State not in nextPossibleStates)) or nextState == currentState:
                    continue
                recompensa = self.reward(nextState)
                curr = self.state_to_tuple(currentState)
                nx = self.state_to_tuple(nextState)
                acciones.append((action,currentState,nextState,recompensa))
                # Initialize nextState in Q-table if not present
                if nx not in self.Q:
                    self.Q[nx] = {a: 0 for a in range(self.actions)}
                self.Q[curr][action] += self.alpha * (recompensa + self.gamma * max(self.Q[nx].values()) - self.Q[curr][action])
                movement = self.getMovement(currentState,nextState)
                self.chess.moveSim(movement[0], movement[1])
                currentState = nextState
            
            if episodio % 500 == 0:
                print("Q-TABLE AL EPISODI ", episodio)
                print(self.Q)
            
            if episodio > 0:
                if self.has_converged(self.Q, self.Q_anterior, umbral_convergencia):
                    convergencias += 1
                    if convergencias >= num_convergencias_necesarias:
                        print(f"Convergencia alcanzada en el episodio {episodio}. Imprimiendo acciones:\n")
                        break
            else:
                convergencias = 0

            # Guardar la Q-table del episodio actual para comparar con la siguiente
            self.Q_anterior = {state: self.Q[state].copy() for state in self.Q}
            

        # Imprimir la Q-table final
        print("\nQ-table final:")
        #print(self.Q)

    def has_converged(self, Q, Q_anterior, umbral_convergencia):
        max_diff = 0
        for state in Q:
            for action in Q[state]:
                if state in Q_anterior and action in Q_anterior[state]:
                    diff = abs(Q[state][action] - Q_anterior[state][action])
                    max_diff = max(max_diff, diff)
                else:
                    max_diff = max(max_diff, abs(Q[state][action]))
        return max_diff < umbral_convergencia


    def getBestState(self, state):
        next_states = self.getListNextStatesW(state)
        max_val = float('-inf')
        best_state = []
        
        for next in next_states:
            val = self.Q[state][0]
            if val > max_val:
                max_val = val
                best_state = next
        return max_val, best_state
        

    def print_action(self, action,current_coords, next_coords, reward):
        # Imprimir la acción y la recompensa
        if action == 0:
            print(f"(Arriba King) --> {next_coords}   Reward: {reward}")
        elif action == 1:
            print(f"(Abajo King) --> {next_coords}   Reward: {reward}")
        elif action == 2:
            print(f"(Izquierda King) --> {next_coords}   Reward: {reward}")
        elif action == 3:
            print(f"(Derecha King) --> {next_coords}   Reward: {reward}")
        elif action == 4:
            print(f"(Diagonal Arriba Izq King) --> {next_coords}   Reward: {reward}")
        elif action == 5:
            print(f"(Diagonal Arriba Der King) --> {next_coords}   Reward: {reward}")
        elif action == 6:
            print(f"(Diagonal Abajo Izq King) --> {next_coords}   Reward: {reward}")
        elif action == 7:
            print(f"(Diagonal Abajo Der King) --> {next_coords}   Reward: {reward}")
        elif action == 8:
            print(f"(Arriba Tower) --> {next_coords}   Reward: {reward}")
        elif action == 9:
            print(f"(Abajo Tower) --> {next_coords}   Reward: {reward}")
        elif action == 10:
            print(f"(Izquierda Tower) --> {next_coords}   Reward: {reward}")
        elif action == 11:
            print(f"(Derecha Tower) --> {next_coords}   Reward: {reward}")


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

    # get list of next states for current state
    print("current State ",currentState,"\n")

    print("Q-table start:\n")
    print(aichess.Q)

    aichess.q_learning_Chess(currentState)
    aichess.print_best_path(currentState)

    print("\nFinal board state\n")
    aichess.chess.boardSim.print_board()

