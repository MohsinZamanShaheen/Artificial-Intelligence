#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue

import chess
import numpy as np
import sys
import heapq
import random
import json

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
        self.total_episodes = 0
        self.actions = 8 + 28
        self.num_filas = 8
        self.num_columnas = 8
        self.alpha = 0.5  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.3  # Exploración-Explotación trade-off
        # Our q-table is of format {state: {action1: q_value, action2: q_value}}
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

        tower_position = self.getPieceState(state, 2)
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

        # check if any of the moving pieces has the same position as black piece.
        if new_state[0][0] == black_king[0][0] and new_state[0][1] == black_king[0][1]:
            return [(tower_position), (king_position)]
        #print("De estado: ", state, " con action ", action, " al estado ", correction_state )

        return new_state    
    
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

    def selection_action_drunken(self, stateActual):
        # Seleccionar una acción basada en epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Exploración aleatoria
            accion = np.random.randint(self.actions)
        else:
            # Explote
            state = self.state_to_tuple(stateActual)
            q_values = self.Q[state]
            accion = max(q_values, key=q_values.get)
        
        # Randomness
        # probabilidad de fallar
        if np.random.rand() < 0.01:
            # Take a random action from all other possibilities
            all_actions = list(range(self.actions))
            all_actions.remove(accion)  # Remove the intended action
            return np.random.choice(all_actions)
        # en el 99% retornara la accion escojida
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

    def sensibleReward(self, state):
        # si checkmante de blancas, recompensa 100
        if self.isCheckMate(state):
            return 100
        # en resto de posiciones usamos lo que proporciona la heurísica
        else:
            return -1 *self.h(state)

    def q_learning_Chess(self, firstState, sensibleReward, drunken_sailor):
        num_episodios = 2000
        umbral_convergencia = 0.0001
        convergencias = 0
        num_convergencias_necesarias = 5

        for episodio in range(num_episodios):
            #print("episodi: ", episodio)
            currentState = firstState
            self.newBoardSim(currentState)
            acciones = []

            while not self.isCheckMate(currentState):
                # select an action. Depends if drunken sailor or normal case(indicated through parameter of this function)
                action = self.selection_action_drunken(currentState) if drunken_sailor else self.selection_action(currentState)

                nextPossibleStates = self.getListNextStatesW(currentState)
                nextState = self.es_accion_valida(currentState,action)
                same_next_State = nextState[::-1] 
                if ((nextState not in nextPossibleStates) and (same_next_State not in nextPossibleStates)) or nextState == currentState:
                    continue
                # depending of which type of reward specified through parameter, one or the other will consideres
                recompensa = self.sensibleReward(nextState) if sensibleReward else self.reward(nextState)
                curr = self.state_to_tuple(currentState)
                nx = self.state_to_tuple(nextState)
                # this is used in case we want to print the actions
                acciones.append((action,currentState,nextState,recompensa))
                # Initialize nextState in Q-table if not present
                if nx not in self.Q:
                    self.Q[nx] = {a: 0 for a in range(self.actions)}
                # update the q_table 
                self.Q[curr][action] += self.alpha * (recompensa + self.gamma * max(self.Q[nx].values()) - self.Q[curr][action])
                # make the movementin the board to the next state
                movement = self.getMovement(currentState,nextState)
                self.chess.moveSim(movement[0], movement[1])
                # update the current state for the next iteration
                currentState = nextState
            
            if episodio % 300 == 0:
                print("\nQ-TABLE AT EPISODE ", episodio)
                # print en formato array/matriz
                self.printQTableAsMatrix(self.Q)
                #Comenta linia anterior y descomente la siguiente si deseas printear en formato original(nested diccionary)
                #print(self.Q)
                #si deseas guardar esta q-table en un fichero
                #self.saveQTableToFile(f'qtableEpisode{episodio}.txt', self.Q)
            
            if episodio > 0:
                if self.has_converged(self.Q, self.Q_anterior, umbral_convergencia):
                    convergencias += 1
                    if convergencias >= num_convergencias_necesarias:
                        self.total_episodes = episodio
                        break
            else:
                convergencias = 0

            # Guardar la Q-table del episodio actual para comparar con la siguiente
            self.Q_anterior = {state: self.Q[state].copy() for state in self.Q}

        # Imprimir la Q-table final
        print("\nQ-table final:")
        self.printQTableAsMatrix(self.Q)
        print()
        #Comenta linia anterior y descomente la siguiente si deseas printear en formato original(nested diccionary)
        #print(self.Q)
        #self.saveQTableToFile('finalQtable.txt', self.Q)

        print(f"\nConvergence achieved in {episodio} episodes.\n")
        # Print actions that lead to checkmate
        for accion, current_coords, next_coords, reward in acciones:
            self.print_action(accion, current_coords, next_coords, reward)

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

    def print_best_path(self, start_state):
        self.newBoardSim(start_state)
        current_state = start_state
        path = [current_state]
        path_found = True

        while not self.isCheckMate(current_state):
            state_key = self.state_to_tuple(current_state) 
            if state_key not in self.Q:
                break
            # Get the action with the highest Q-value for the current state
            best_action = max(self.Q[state_key], key=self.Q[state_key].get)
            next_state = self.es_accion_valida(current_state,best_action)
            if current_state != next_state:  
                movement = self.getMovement(current_state, next_state)
                self.chess.moveSim(movement[0], movement[1])
            else:
                path_found = False
                print("Could not trace path, agent might need further learning. !!Hint!! Consider retrying with different parametres. ")
                break

            current_state = next_state
            path.append(current_state)

        if path_found:
            # Print the final path
            print("\nBest path to checkmate is:")
            print(path)  

    def printQTableAsMatrix(self,Q):

        # Create mappings for states and actions
        state_to_index = {state: i for i, state in enumerate(sorted(Q.keys()))}
        action_to_index = {action: j for j, action in enumerate(sorted(set(action for actions in Q.values() for action in actions)))}

        # Initialize the Q-table as matrix
        num_states = len(state_to_index)
        num_actions = len(action_to_index)
        Q_matrix = np.zeros((num_states, num_actions))

        # Fill the matrix with Q-values
        for state, actions in Q.items():
            state_idx = state_to_index[state]
            for action, q_value in actions.items():
                action_idx = action_to_index[action]
                Q_matrix[state_idx, action_idx] = q_value

        # Print the Q-table in matrix format
        print("\nQ-Table:")
        print(Q_matrix)

    # This function is to save the q-table which is in dictonary format in a file
    def saveQTableToFile(self,filename, Q):
        Q_serializable = {str(state): actions for state, actions in Q.items()}
        # Save the Q-table to a text file
        with open(filename, 'w') as file:
            json.dump(Q_serializable, file, indent=4)

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
        elif action == 8 or action == 9 or action == 19 or action == 11 or action == 12 or action == 13 or action == 14:
            print(f"(Arriba Tower) --> {next_coords}   Reward: {reward}")
        elif action == 15 or action == 16 or action == 17 or action == 18 or action == 19 or action == 20 or action == 21:
            print(f"(Abajo Tower) --> {next_coords}   Reward: {reward}")
        elif action == 22 or action == 23 or action == 24 or action == 25 or action == 26 or action == 27 or action == 28:
            print(f"(Izquierda Tower) --> {next_coords}   Reward: {reward}")
        elif action == 29 or action == 30 or action == 31 or action == 32 or action == 33 or action == 34 or action == 35:
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
    #aichess.printQTableAsMatrix(aichess.Q)
    #DeComenta linia anterior y comenta la siguiente si deseas printear en formato numpy matrix
    # esto imprime formato original (nested dictionary)
    print(aichess.Q)

    # This is to save the initial q-table to a file 
    #aichess.saveQTableToFile('initialQtable.txt', aichess.Q)

    # For section (2a) use sensibleReward = False
    sensibleReward = False
    # For section (2b) use sensibleReward = True
    #sensibleReward = True

    # For (2c) use drunken_sailor = True
    drunken_sailor = False
    #drunken_sailor = True
    aichess.q_learning_Chess(currentState,sensibleReward,drunken_sailor)
    aichess.print_best_path(currentState)

    print("\nFinal board state\n")
    aichess.chess.boardSim.print_board()

