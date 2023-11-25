import numpy as np
import matplotlib.pyplot as plt
from typing import List
RawStateType = List[List[List[int]]]

from itertools import permutations

class Scenario():
    def __init__(self, apartado):
        self.num_rows = 3
        self.num_columns = 4
        self.num_states = self.num_rows * self.num_columns
        self.num_actions = 4
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.gamma = 0  # Factor de descuento
        self.alpha = 0 # Tasa de aprendizaje
        self.start_position = (2, 0)
        self.goal_position = (0, 3)
        self.goal_state = self.goal_position[0] * self.num_columns + self.goal_position[1]
        self.blocked_position = (1, 1)
        self.blocked_state = self.blocked_position[0] * self.num_columns + self.blocked_position[1]
        if apartado == 'everywhere':
            self.rewards = np.full((self.num_states, self.num_actions),
                                   -1)  # Todas las acciones tienen una recompensa de -1
            # La acción que lleva a la casilla bloqueada tiene una recompensa de -10
            self.rewards[self.blocked_state, :] = -10
            # La acción que lleva al estado final tiene una recompensa de 100
            self.rewards[self.goal_state, :] = 100

        elif apartado == 'custom_rewards':
            self.startReward = -5
            self.blockedReward = -10
            self.goalReward = 100
            self.rewards = np.zeros((self.num_states, self.num_actions))

            # Asignar manualmente las recompensas según la estructura proporcionada
            self.rewards[0, :] = [-3, -3, -3, -3]
            self.rewards[1, :] = [-2, -2, -2, -2]
            self.rewards[2, :] = [-1, -1, -1, -1]
            self.rewards[3, :] = [self.goalReward, self.goalReward, self.goalReward, self.goalReward]
            self.rewards[4, :] = [-4, -4, -4, -4]
            self.rewards[5, :] = [self.blockedReward, self.blockedReward, self.blockedReward, self.blockedReward]
            self.rewards[6, :] = [-2, -2, -2, -2]
            self.rewards[7, :] = [-1, -1, -1, -1]
            self.rewards[8, :] = [self.startReward, self.startReward, self.startReward, self.startReward]
            self.rewards[9, :] = [-4, -4, -4, -4]
            self.rewards[10, :] = [-3, -3, -3, -3]
            self.rewards[11, :] = [-2, -2, -2, -2]


        self.num_episodes = 1000

    def visualize_board(self, optimal_path_coords):
        board = np.zeros((self.num_rows, self.num_columns), dtype=int)

        for coord in optimal_path_coords:
            board[coord[0], coord[1]] = 1

        board[self.blocked_position[0], self.blocked_position[1]] = -1  # Marcamos la casilla bloqueada en rojo
        plt.text(self.blocked_position[1], self.blocked_position[0], "BLOCKED", ha='center', va='center', fontsize=8)

        board[self.start_position[0], self.start_position[1]] = 2  # Etiquetamos la posición de inicio
        plt.text(self.start_position[1], self.start_position[0], "START", ha='center', va='center', fontsize=8)

        board[self.goal_position[0], self.goal_position[1]] = 3  # Etiquetamos la posición objetivo
        plt.text(self.goal_position[1], self.goal_position[0], "GOAL", ha='center', va='center', fontsize=8)



        plt.imshow(board, cmap='RdYlBu', origin='upper', interpolation='none')
        x_ticks_labels = [str(label) for label in range(1, self.num_columns + 1)]
        y_ticks_labels = [str(label) for label in range(self.num_rows, 0, -1)]

        plt.xticks(range(self.num_columns), x_ticks_labels)
        plt.yticks(range(self.num_rows), y_ticks_labels)
        plt.title('Path Q-Learning')
        plt.show()

    def select_action(self, state):
        # Seleccionar la acción con mayor valor en la matriz Q para el estado dado
        return np.argmax(self.Q[state, :])

    def move_action(self, current_state, action):
        if action == 0 and current_state >= self.num_columns:
            return current_state - self.num_columns
        elif action == 1 and current_state < self.num_states - self.num_columns:
            return current_state + self.num_columns
        elif action == 2 and current_state % self.num_columns != 0:
            return current_state - 1
        elif action == 3 and (current_state + 1) % self.num_columns != 0:
            return current_state + 1
        else:
            return current_state

    def q_learning(self):
        print_q_interval = self.num_episodes / 2
        for episode in range(self.num_episodes):

            # Inicializar el estado inicial
            current_state = self.num_columns * (self.num_rows - 1)  # Start position


            while current_state != self.goal_state:  # Mientras no estemos en el estado final
                # Seleccionar una acción
                action = self.select_action(current_state)

                # Obtener la recompensa para la acción tomada
                reward = self.rewards[current_state, action]

                # Moverse al siguiente estado
                next_state = self.move_action(current_state, action)



                # Actualizar la matriz Q usando la ecuación de Q-learning
                self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * (
                            reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state, action])

                # Actualizar el estado actual al siguiente estado
                current_state = next_state

            # Imprimir la matriz Q en intervalos específicos
            if episode == print_q_interval or episode == 1:
                    print(f"\nQ-table (Episode {episode}):")
                    print(self.Q)

        # Imprimir la matriz Q
        print("Q-table Final:")
        print(self.Q)

        # Imprimir el camino óptimo
        optimal_path = [self.num_columns * (self.num_rows - 1)]  # Start position
        current_state = optimal_path[0]

        print("ACTIONS:")
        while current_state != self.goal_state:
            action = self.select_action(current_state)

            next_state = self.move_action(current_state, action)
            
            # Convertir estados a coordenadas (filas, columnas)
            current_coords = (current_state // self.num_columns, current_state % self.num_columns)
            next_coords = (next_state // self.num_columns, next_state % self.num_columns)

            # Imprimir la acción
            if action == 0:
                print(current_coords, " (Arriba) --> ", next_coords)
            elif action == 1:
                print(current_coords, " (Abajo) --> ", next_coords)
            elif action == 2:
                print(current_coords, " (Izquierda) --> ", next_coords)
            elif action == 3:
                print(current_coords, " (Derecha) --> ", next_coords)

            optimal_path.append(next_state)
            current_state = next_state

        # Convertir estados a coordenadas (filas, columnas) para el estado final
        optimal_path_coords = [(state // self.num_columns, state % self.num_columns) for state in optimal_path]


        return optimal_path_coords

    def q_learning_drunken_sailor(self):
        print_q_interval = self.num_episodes / 2
        for episode in range(self.num_episodes):

            # Inicializar el estado inicial
            current_state = self.num_columns * (self.num_rows - 1)  # Start position

            while current_state != self.goal_state:  # Mientras no estemos en el estado final
                # Seleccionar una acción
                if np.random.rand() < 0.99:
                    action = self.select_action(current_state)
                else:
                    action = np.random.randint(self.num_actions)

                # Obtener la recompensa para la acción tomada
                reward = self.rewards[current_state, action]

                # Moverse al siguiente estado
                next_state = self.move_action(current_state, action)

                # Actualizar la matriz Q usando la ecuación de Q-learning
                self.Q[current_state, action] = self.Q[current_state, action] + self.alpha * (
                        reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[current_state, action])

                # Actualizar el estado actual al siguiente estado
                current_state = next_state

            # Imprimir la matriz Q en intervalos específicos
            if episode == print_q_interval or episode == 1:
                print(f"\nQ-table (Episode {episode}):")
                print(self.Q)

        # Imprimir la matriz Q
        print("Q-table Final:")
        print(self.Q)

        # Imprimir el camino óptimo
        optimal_path = [self.num_columns * (self.num_rows - 1)]  # Start position
        current_state = optimal_path[0]

        print("ACTIONS:")
        while current_state != self.goal_state:
            action = self.select_action(current_state)

            next_state = self.move_action(current_state, action)

            # Convertir estados a coordenadas (filas, columnas)
            current_coords = (current_state // self.num_columns, current_state % self.num_columns)
            next_coords = (next_state // self.num_columns, next_state % self.num_columns)

            # Imprimir la acción
            if action == 0:
                print(current_coords, " (Arriba) --> ", next_coords)
            elif action == 1:
                print(current_coords, " (Abajo) --> ", next_coords)
            elif action == 2:
                print(current_coords, " (Izquierda) --> ", next_coords)
            elif action == 3:
                print(current_coords, " (Derecha) --> ", next_coords)

            optimal_path.append(next_state)
            current_state = next_state

        # Convertir estados a coordenadas (filas, columnas) para el estado final
        optimal_path_coords = [(state // self.num_columns, state % self.num_columns) for state in optimal_path]

        return optimal_path_coords

if __name__ == "__main__":
    #Para cambiar a las rewards diferentes, descomentar codigo de abajo, y comentar 'everywhere'
    #Si quereis el -1 en todos lados, y no el custom, comentar custom, y descomentar everywhere
    #apartado = 'everywhere'
    apartado = 'custom_rewards'

    #Si quereis que el agente tenga total eleccion de sus acciones, dejar descomentado = False,
    #En caso contrario, dejar descomentado el =True
    #drunken_sailor = False
    drunken_sailor = True
    scenario = Scenario(apartado)
    print("Q-table (Initial):")
    print(scenario.Q)
    scenario.gamma = 0.8  # Factor de descuento
    scenario.alpha = 0.1  # Tasa de aprendizaje
    if drunken_sailor:
        optimal_path = scenario.q_learning_drunken_sailor()
    else:
        optimal_path = scenario.q_learning()
    scenario.visualize_board(optimal_path)