import numpy as np

class Scenario:
    def __init__(self, rewards_type):
        self.num_filas = 3
        self.num_columnas = 4
        self.num_acciones = 4
        self.alpha = 0.1  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.1  # Exploración-Explotación trade-off
        self.bloqueados = [(1, 1)]
        self.rewards_type = rewards_type

        if self.rewards_type == 'everywhere':
            self.recompensa_goal = 100
            self.recompensa_default = -1
        elif self.rewards_type == 'custom_rewards':
            self.recompensas = np.array([[-3, -2, -1, 100],
                                         [-4, 0, -2, -1],
                                         [-5, -4, -3, -2]])
        else:
            raise ValueError("Tipo de recompensas no válido.")

        self.Q = np.zeros((self.num_filas * self.num_columnas, self.num_acciones))


    def obtener_estado(self, fila, columna):
        return fila * self.num_columnas + columna

    def es_accion_valida(self, fila, columna, accion):
        nueva_fila, nueva_columna = fila, columna

        if accion == 0:  # Mover hacia arriba
            nueva_fila = max(fila - 1, 0)
        elif accion == 1:  # Mover hacia abajo
            nueva_fila = min(fila + 1, self.num_filas - 1)
        elif accion == 2:  # Mover hacia la izquierda
            nueva_columna = max(columna - 1, 0)
        elif accion == 3:  # Mover hacia la derecha
            nueva_columna = min(columna + 1, self.num_columnas - 1)

        # Verificar si la nueva posición está bloqueada
        if (nueva_fila, nueva_columna) in self.bloqueados:
            return (fila, columna)

        return (nueva_fila, nueva_columna)

    def selection_action(self, fila_actual, columna_actual):
        # Seleccionar una acción basada en epsilon-greedy
        if np.random.rand() < self.epsilon:
            accion = np.random.randint(self.num_acciones)  # Exploración aleatoria
        else: ## caso de explotación
            accion = np.argmax(self.Q[self.obtener_estado(fila_actual, columna_actual)])
        return accion

    def q_learning(self, startPosition, goalPosition):
        num_episodios = 1000
        umbral_convergencia = 0.01
        convergencias = 0
        num_convergencias_necesarias = 5

        for episodio in range(num_episodios):
            fila_actual, columna_actual = startPosition  # Corresponde al (1,1) de la tabla del enunciado
            acciones = []

            # mientras no sea estado terminal
            while not (fila_actual, columna_actual) == goalPosition:
                
                accion = self.selection_action(fila_actual, columna_actual)

                # Realizar la acción y obtener la nueva posición
                nueva_fila, nueva_columna = self.es_accion_valida(fila_actual, columna_actual, accion)

                if self.rewards_type == 'everywhere':
                    recompensa = self.recompensa_goal if (nueva_fila, nueva_columna) == (
                    0, 3) else self.recompensa_default
                elif self.rewards_type == 'custom_rewards':
                    recompensa = self.recompensas[nueva_fila, nueva_columna]
                else:
                    raise ValueError("Tipo de recompensas no válido.")

                acciones.append((accion, (fila_actual, columna_actual), (nueva_fila, nueva_columna), recompensa))

                # Actualizar la Q-table usando la ecuación de Bellman
                nuevo_estado = self.obtener_estado(nueva_fila, nueva_columna)
                estado_actual = self.obtener_estado(fila_actual, columna_actual)
                self.Q[estado_actual, accion] += self.alpha * (recompensa + self.gamma * np.max(self.Q[nuevo_estado]) - self.Q[estado_actual, accion])

                # Actualizar la posición actual
                fila_actual, columna_actual = nueva_fila, nueva_columna
            

            # Verificar convergencia
            if episodio > 0 and np.max(np.abs(self.Q - self.Q_anterior)) < umbral_convergencia:
                convergencias += 1
                if convergencias >= num_convergencias_necesarias:
                    print(f"Convergencia alcanzada en el episodio {episodio}. Imprimiendo acciones:\n")
                    for accion, current_coords, next_coords, reward in acciones:
                        self.print_action(accion, current_coords, next_coords, reward)
                    break
            else:
                convergencias = 0

            # Guardar la Q-table del episodio actual para comparar con la siguiente
            self.Q_anterior = np.copy(self.Q)

        # Imprimir la Q-table final
        print("\nQ-table final:")
        print(self.Q)

    
    def selection_action_drunken(self, fila, columna, accion_elegida):
        # Probabilidad de fallar el movimiento
        if np.random.rand() < 0.01:
            # Devolver otra posible acción que no sea la escogida
            posibles_acciones = [a for a in range(self.num_acciones) if a != accion_elegida]
            return np.random.choice(posibles_acciones)

        # Seleccionar la acción escogida
        return accion_elegida

    def q_learning_drunken_sailor(self, startPosition, goalPosition):
        num_episodios = 1000
        umbral_convergencia = 0.0001
        convergencias = 0
        num_convergencias_necesarias = 5

        for episodio in range(num_episodios):
            fila_actual, columna_actual = startPosition  # Iniciar en la esquina superior izquierda
            acciones = []

            while not (fila_actual, columna_actual) == goalPosition:
                possible_action = self.selection_action(fila_actual, columna_actual)
                accion = self.selection_action_drunken(fila_actual, columna_actual, possible_action)

                # Verificar si el movimiento fue exitoso
                if accion is not None:
                    # Realizar la acción y obtener la nueva posición
                    nueva_fila, nueva_columna = self.es_accion_valida(fila_actual, columna_actual, accion)

                    if self.rewards_type == 'everywhere':
                        recompensa = self.recompensa_goal if (nueva_fila, nueva_columna) == (
                            0, 3) else self.recompensa_default
                    elif self.rewards_type == 'custom_rewards':
                        recompensa = self.recompensas[nueva_fila, nueva_columna]
                    else:
                        raise ValueError("Tipo de recompensas no válido.")

                    acciones.append((accion, (fila_actual, columna_actual), (nueva_fila, nueva_columna), recompensa))

                    # Actualizar la Q-table usando la ecuación de Bellman
                    nuevo_estado = self.obtener_estado(nueva_fila, nueva_columna)
                    self.Q[self.obtener_estado(fila_actual, columna_actual), accion] += \
                        self.alpha * (recompensa + self.gamma * np.max(self.Q[nuevo_estado]) - self.Q[
                            self.obtener_estado(fila_actual, columna_actual), accion])

                    # Actualizar la posición actual
                    fila_actual, columna_actual = nueva_fila, nueva_columna
                else:
                    continue

            # Verificar convergencia
            if episodio > 0 and np.max(np.abs(self.Q - self.Q_anterior)) < umbral_convergencia:
                convergencias += 1
                if convergencias >= num_convergencias_necesarias:
                    print(f"Convergencia alcanzada en el episodio {episodio}. Imprimiendo acciones:\n")
                    for accion, current_coords, next_coords, reward in acciones:
                        self.print_action(accion, current_coords, next_coords, reward)
                    break
            else:
                convergencias = 0

            # Guardar la Q-table del episodio actual para comparar con la siguiente
            self.Q_anterior = np.copy(self.Q)

        # Imprimir la Q-table final
        print("\nQ-table final:")
        print(self.Q)

    
    def print_action(self, action,current_coords, next_coords, reward):
        # Imprimir la acción y la recompensa
        if action == 0:
            print(f"(Arriba) --> {next_coords}   Reward: {reward}")
        elif action == 1:
            print(f"(Abajo) --> {next_coords}   Reward: {reward}")
        elif action == 2:
            print(f"(Izquierda) --> {next_coords}   Reward: {reward}")
        elif action == 3:
            print(f"(Derecha) --> {next_coords}   Reward: {reward}")

        self.print_board(next_coords)
        print()


    def print_board(self, coords):
        print("+" + "-" * (self.num_columnas * 4 - 1) + "+")
        for fila in range(self.num_filas):
            print("|", end=' ')
            for columna in range(self.num_columnas):
                if (fila, columna) == coords:
                    print('O', end=' ')
                elif (fila, columna) in self.bloqueados:
                    print('X', end=' ')
                else:
                    print('-', end=' ')
                print("|", end=' ')
            print()
        print("+" + "-" * (self.num_columnas * 4 - 1) + "+")
        


if __name__ == "__main__":
    # Para cambiar a las rewards diferentes, descomentar codigo de abajo, y comentar 'everywhere'
    # Si quereis el -1 en todos lados, y no el custom, comentar custom, y descomentar everywhere
    #apartado = 'everywhere'
    apartado = 'custom_rewards'

    # Si quereis que el agente tenga total eleccion de sus acciones, dejar descomentado = False,
    # En caso contrario, dejar descomentado el =True
    #drunken_sailor = False
    drunken_sailor = True
    scenario = Scenario(apartado)
    print("\nQ-table (Initial):")
    print(scenario.Q)
    scenario.gamma = 0.8  # Factor de descuento
    scenario.alpha = 0.1  # Tasa de aprendizaje

    start = (2,0)
    goal = (0,3)
    print("\nBoard inicial")
    scenario.print_board(start)
    
    if drunken_sailor:
        scenario.q_learning_drunken_sailor(start,goal)
    else:
        scenario.q_learning(start,goal)
