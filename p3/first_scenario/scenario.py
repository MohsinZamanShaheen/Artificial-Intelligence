import numpy as np
import matplotlib.pyplot as plt
from typing import List
RawStateType = List[List[List[int]]]

from itertools import permutations

class Scenario():
    def __init__(self, apartado):


        self.num_filas = 3
        self.num_columnas = 4
        self.num_acciones = 4
        self.Q = np.zeros((self.num_filas, self.num_columnas, self.num_acciones))


        self.gamma = 0  # Factor de descuento
        self.alpha = 0 # Tasa de aprendizaje
        self.num_episodios = 1000
        self.estado_bloqueado = (1, 1)
        self.estado_inicio = (0, 0)
        self.estado_final = (2, 3)
        self.epsilon = 0.1  # Exploración inicial
        if apartado == 'custom_rewards':
            # Inicializar recompensas personalizadas
            self.recompensas = np.array([[-5, -4, -3, -2],
                                         [-4, 0, -2, -1],
                                         [-3, -2, -1, 100]])
        elif apartado == 'everywhere':
            # Utilizar recompensas en todas partes por defecto
            self.recompensas = -1 * np.ones((self.num_filas, self.num_columnas))
            # Establecer 100 en el estado objetivo
            self.recompensas[self.estado_final[0], self.estado_final[1]] = 100


    def seleccionar_accion(self, q_values, estado, epsilon=0.1):
        acciones_posibles = [a for a in range(self.num_acciones) if
                             self.estado_siguiente(estado, a) != self.estado_bloqueado]

        if np.random.rand() < epsilon:
            return np.random.choice(acciones_posibles)
        else:
            return np.argmax(q_values[estado][acciones_posibles])

    # Función para obtener el próximo estado dado un estado y una acción
    def estado_siguiente(self, estado, accion):
        next_state = None
        if accion == 0:  # Abajo
            next_state = (max(estado[0] - 1, 0), estado[1])
        elif accion == 1:  # Arriba
            next_state = (min(estado[0] + 1, self.num_filas - 1), estado[1])
        elif accion == 2:  # Izquierda
            next_state = (estado[0], max(estado[1] - 1, 0))
        elif accion == 3:  # Derecha
            next_state = (estado[0], min(estado[1] + 1, self.num_columnas - 1))

        # Verificar si el próximo estado es el estado bloqueado y ajustar
        if next_state == self.estado_bloqueado:
            return estado
        else:
            return next_state

    def printAction(self, action, current_coords, next_coords):
        # Imprimir la acción
        if action == 0:
            print(current_coords, " (Abajo) --> ", next_coords)
        elif action == 1:
            print(current_coords, " (Arriba) --> ", next_coords)
        elif action == 2:
            print(current_coords, " (Izquierda) --> ", next_coords)
        elif action == 3:
            print(current_coords, " (Derecha) --> ", next_coords)

    def visualize_board(self, optimal_path_coords, blocked_position, start_position, goal_position, num_rows, num_columns):
        board = np.zeros((num_rows, num_columns), dtype=int)

        for coord in optimal_path_coords:
            inverted_coord = (num_rows - 1 - coord[0], coord[1])
            board[inverted_coord[0], inverted_coord[1]] = 1

        inverted_blocked_position = (num_rows - 1 - blocked_position[0], blocked_position[1])
        board[inverted_blocked_position[0], inverted_blocked_position[1]] = -1  # Marcamos la casilla bloqueada en rojo
        plt.text(inverted_blocked_position[1], inverted_blocked_position[0], "BLOCKED", ha='center', va='center',
                 fontsize=8)

        inverted_start_position = (num_rows - 1 - start_position[0], start_position[1])
        board[inverted_start_position[0], inverted_start_position[1]] = 2  # Etiquetamos la posición de inicio
        plt.text(inverted_start_position[1], inverted_start_position[0], "START", ha='center', va='center', fontsize=8)

        inverted_goal_position = (num_rows - 1 - goal_position[0], goal_position[1])
        board[inverted_goal_position[0], inverted_goal_position[1]] = 3  # Etiquetamos la posición objetivo
        plt.text(inverted_goal_position[1], inverted_goal_position[0], "GOAL", ha='center', va='center', fontsize=8)

        plt.imshow(board, cmap='RdYlBu', origin='upper', interpolation='none')
        x_ticks_labels = [str(label) for label in range(1, num_columns + 1)]
        y_ticks_labels = [str(label) for label in range(num_rows, 0, -1)]  # Invertir las etiquetas del eje y

        plt.xticks(range(num_columns), x_ticks_labels)
        plt.yticks(range(num_rows), y_ticks_labels)
        plt.title('Path Q-Learning')
        plt.show()

    def visualize_board_drunken(self, optimal_path_coords, blocked_position, start_position, goal_position, num_rows,
                                num_columns):
        board = np.zeros((num_rows, num_columns), dtype=int)

        for coord in optimal_path_coords:
            if coord is not None:
                inverted_coord = (num_rows - 1 - coord[0], coord[1])
                board[inverted_coord[0], inverted_coord[1]] = 1

        if blocked_position is not None:
            inverted_blocked_position = (num_rows - 1 - blocked_position[0], blocked_position[1])
            board[inverted_blocked_position[0], inverted_blocked_position[
                1]] = -1  # Marcamos la casilla bloqueada en rojo
            plt.text(inverted_blocked_position[1], inverted_blocked_position[0], "BLOCKED", ha='center', va='center',
                     fontsize=8)

        if start_position is not None:
            inverted_start_position = (num_rows - 1 - start_position[0], start_position[1])
            board[inverted_start_position[0], inverted_start_position[1]] = 2  # Etiquetamos la posición de inicio
            plt.text(inverted_start_position[1], inverted_start_position[0], "START", ha='center', va='center',
                     fontsize=8)

        if goal_position is not None:
            inverted_goal_position = (num_rows - 1 - goal_position[0], goal_position[1])
            board[inverted_goal_position[0], inverted_goal_position[1]] = 3  # Etiquetamos la posición objetivo
            plt.text(inverted_goal_position[1], inverted_goal_position[0], "GOAL", ha='center', va='center', fontsize=8)

        plt.imshow(board, cmap='RdYlBu', origin='upper', interpolation='none')
        x_ticks_labels = [str(label) for label in range(1, num_columns + 1)]
        y_ticks_labels = [str(label) for label in range(num_rows, 0, -1)]  # Invertir las etiquetas del eje y

        plt.xticks(range(num_columns), x_ticks_labels)
        plt.yticks(range(num_rows), y_ticks_labels)
        plt.title('Path Q-Learning - Drunken Sailor')
        plt.show()

    def q_learning(self):
        for episodio in range(self.num_episodios):
            estado_actual = (0, 0)  # Estado inicial
            acciones_realizadas = []  # Lista para almacenar las acciones del episodio actual
            estados = []

            while estado_actual != self.estado_final:
                # Elegir una acción
                accion = self.seleccionar_accion(self.Q, estado_actual, self.epsilon)
                estados.append(estado_actual)

                # Tomar la acción y obtener la recompensa
                estado_siguiente_val = self.estado_siguiente(estado_actual, accion)
                if estado_siguiente_val != self.estado_bloqueado:
                    recompensa = self.recompensas[estado_actual[0], estado_actual[1]]


                # Actualizar la tabla Q usando la ecuación de Q-learning
                self.Q[estado_actual][accion] += self.alpha * (
                            recompensa + self.gamma * np.max(self.Q[estado_siguiente_val]) - self.Q[estado_actual][accion])

                acciones_realizadas.append((accion, estado_actual, estado_siguiente_val))  # Almacenar la acción tomada
                # Actualizar el estado actual
                estado_actual = estado_siguiente_val

            # Reducir la exploración a medida que avanzan los episodios
            self.epsilon = max(0.1, self.epsilon * 0.99)

            # Imprimir las acciones realizadas en el episodio actual
            if episodio == self.num_episodios - 1:
                print(f"\nAcciones realizadas en el episodio {episodio + 1}:")
                for accion, current, next_coords in acciones_realizadas:
                    self.printAction(accion, current, next_coords)
                # Visualizar el tablero con el camino óptimo
                self.visualize_board(estados, self.estado_bloqueado, self.estado_inicio, self.estado_final,
                                     self.num_filas, self.num_columnas)  # Ajustado para comenzar en (1, 1)

        # Imprimir la tabla Q aprendida
        print("\nTabla Q aprendida:")
        print(self.Q)

        return 0

    def seleccionar_accion_drunken_sailor(self, q_values, estado, epsilon=0.1, prob_fallo=0.01):
        # Introducir el 1% de probabilidad de que no se realice ningún movimiento
        if np.random.rand() < prob_fallo:
            return None
        else:
            if np.random.rand() < epsilon:
                # Seleccionar una acción aleatoria excluyendo acciones que lleven al estado bloqueado
                acciones_posibles = [a for a in range(self.num_acciones) if
                                     self.estado_siguiente(estado, a) != self.estado_bloqueado]
                return np.random.choice(acciones_posibles)
            else:
                # Seleccionar la acción que maximiza Q, excluyendo acciones que lleven al estado bloqueado
                acciones_posibles = [a for a in range(self.num_acciones) if
                                     self.estado_siguiente(estado, a) != self.estado_bloqueado]
                return np.argmax(q_values[estado][acciones_posibles])

    # Modificar la función q_learning_drunken_sailor para usar la nueva función de selección
    def q_learning_drunken_sailor(self):
        for episodio in range(self.num_episodios):
            estado_actual = (0, 0)  # Estado inicial
            acciones_realizadas = []  # Lista para almacenar las acciones del episodio actual
            estados = []

            while estado_actual != self.estado_final:
                # Elegir una acción
                accion = self.seleccionar_accion_drunken_sailor(self.Q, estado_actual, self.epsilon)

                if accion is not None:
                    # Tomar la acción y obtener la recompensa
                    estado_siguiente_val = self.estado_siguiente(estado_actual, accion)
                    if estado_siguiente_val != self.estado_bloqueado:
                        recompensa = self.recompensas[estado_actual[0], estado_actual[1]]

                    # Actualizar la tabla Q usando la ecuación de Q-learning
                    self.Q[estado_actual][accion] += self.alpha * (
                            recompensa + self.gamma * np.max(self.Q[estado_siguiente_val]) - self.Q[estado_actual][
                        accion])

                    acciones_realizadas.append(
                        (accion, estado_actual, estado_siguiente_val))  # Almacenar la acción tomada
                    # Actualizar el estado actual
                    estado_actual = estado_siguiente_val
                else:
                    pass
            # Reducir la exploración a medida que avanzan los episodios
            self.epsilon = max(0.1, self.epsilon * 0.99)

            # Imprimir las acciones realizadas en el episodio actual
            if episodio == self.num_episodios - 1:

                print(f"\nAcciones realizadas en el episodio {episodio + 1}:")
                for accion, current, next_coords in acciones_realizadas:
                    if accion is not None:
                        self.printAction(accion, current, next_coords)
                if accion is not None:
                    # Visualizar el tablero con el camino óptimo
                    self.visualize_board_drunken(estados, self.estado_bloqueado, self.estado_inicio, self.estado_final,
                                         self.num_filas, self.num_columnas)  # Ajustado para comenzar en (1, 1)

        # Imprimir la tabla Q aprendida
        print("\nTabla Q aprendida:")
        print(self.Q)

        return 0


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
        scenario.q_learning_drunken_sailor()
    else:
        scenario.q_learning()