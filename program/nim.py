import math
import random
import time


class Nim():

    def __init__(self, initial = [1, 3, 5, 7]):

        '''
            Initialize game board.
            Each game board has
                - 'piles'  : a list of how many elements remain in each pile
                - 'player' : 0 or 1 to indicate which player's turn
                - 'winner' : None, 0, or 1 to indicate who the winner is
        '''
        
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    # Alteração: avaliable_action -> available_actions
    # Isso foi feito porque ele estava sendo instânciado com o nome correto
    # Como ele estava com o nome errado ele não era reconhecido, um erro bobo
    def available_actions(self, piles):
        
        '''
            self.avaliable_actions(piles) takes a 'piles' list as input
            and returns all of the available actions '(i, j)' int that state.

            Action '(i, j)' represents the action of removing 'j' items
            from pile 'i'.
        '''

        actions = set()

        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))

        return actions

    def other_player(self, player):

        '''
            self.other_player(plauyer) returns the player that is not
            'player'. Assumes 'player' is either 0 or 1.
        '''

        return 0 if player == 1 else 1

    def switch_player(self):

        '''
            Switch the current player to the other player.
        '''

        self.player = self.other_player(self.player)

    def move(self, action):
        
        '''
            Make the move 'action' for the current player.
            'action' must be a tuple '(i, j)'.
        '''

        pile, count = action

        # check for errors
        if self.winner is not None:
            raise Exception('Game already won.')
        else:
            if pile < 0 or pile >= len(self.piles):
                raise Exception('Invalid pile.')
            else:
                if count < 1 or count > self.piles[pile]:
                    raise Exception('Invalid number of objects.')
                
        # update pile
        self.piles[pile] -= count
        self.switch_player()

        # check the winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class SARSA():

    '''
        DIFERENÇAS ENTRE SARSA E QLEARNING

        - QLearning é Off-Policy, logo ele trabalha com a ação que o agente ESCOLHERIA, prevendo o futuro
        e escolhendo uma ação baseada na previsão
        - Sarsa é On-Policy, logo ele trabalha com a ação que o agente ESCOLHEU, depois ele atualiza as
        tabelas sem necessidade de prever futuro algum

        RACÍOCINIO EM ETAPAS DO SARSA

        1.Cria uma matriz Q que irá receber valores de recompensa baseados em uma exploração aleatória, 
        executando uma ação e recebendo uma recompensa
        2.O agente no estado s0, escolhe e realiza uma ação inicial aleatória a0, pega sua recomepensa 
        e vai para o próximo estado s1
        3.Ele repete a escolha porém para a ação a1
        4.Agora com as ações a0 e a1 e os estados s0 e s1 ele atualiza a matriz Q
        5.Cálculo: "Q[s0][a0] = Q[s0][a0] + alpha*(r + gamma*Q[s1][a1] - Q[s0][a0])"
        6.Repete tentando melhorar a cada resultado
    '''

    # Alteração: Adicionado o Gamma pelo mesmo motivo do QLearning
    def __init__(self, alpha = 0.5, epsilon = 0.1, gamma = 0.9):

        '''
            Initialize AI with an empty SARSA dictionary,
            an alpha rate and an epsilon rate.

            The SARSA dictionary maps '(state, action)
            pairs to a Q-value.
                - 'state' is a tuple of remaining piles, e.g. [1, 1, 4, 4]
                - 'action' is a tuple '(i, j)' for an action
        '''

        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def update_model(self, old_state, action, new_state, reward):

        '''
            Update SARSA model, given and old state, an action taken
            in that state, a new resulting state, and the reward received
            from taking that action.
        '''

        old_q = self.get_value(old_state, action)
        # OBSERVAÇÃO: Aqui é onde o SARSA se diferencia do QLearning (além do método choose_actions)
        new_action = self.choose_action(new_state)
        best_future = self.get_value(new_state, new_action)
        self.update_value(old_state, action, old_q, reward, best_future)

    def get_value(self, state, action):

        '''
            Return the Q-value for the state 'state' and the action 'action'.
            If no Q-value exists yet in 'self.q', return 0.
        '''

        value = self.q.get((tuple(state), action), 0)   
        return value

    def update_value(self, old_state, action, old_q, reward, future_rewards):

        '''
            Update the Q-value for the state 'state' and the action 'action'
            given the previous Q-value 'old_q', a current reward 'reward',
            and an estimate of future rewards 'future_rewards'.
        '''

        new_value = old_q + self.alpha * (reward + self.gamma * future_rewards - old_q)
        self.q[tuple(old_state), action] = new_value

    def choose_action(self, state, epsilon = True):

        '''
            Given a state 'state', return a action '(i, j)' to take.
            
            If 'epsilon' is 'False', then return the best action
            avaiable in the state (the one with the highest Q-value, 
            using 0 for pairs that have no Q-values).

            If 'epsilon' is 'True', then with probability 'self.epsilon'
            chose a random available action, otherwise chose the best
            action available.

            If multiple actions have the same Q-value, any of those
            options is an acceptable return value.
        '''

        # OBSERVAÇÃO: aqui o SARSA também se diferencia do QLearning, sendo até mais simples já que a lógica é
        # em resumo, escolher uma ação aleatória e anotar o resultado até achar as melhores ações para cada
        # situação que o jogo possa te jogar
        game = Nim()
        actions = list(game.available_actions(state))
        if not actions:
            return 0

        if epsilon:
            if random.random() < self.epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda action: self.get_value(state, action))
        else:
            action = max(actions, key=lambda action: self.get_value(state, action))

        return action
        

class QLearning():

    '''
        RACÍOCINIO EM ETAPAS DO Q-LEARNING

        1.Cria uma matriz Q com valores de recompensa **PREVISTOS** para cada [estado][ação]
        2.Recalcula o valor de cada recompensa baseado no resultado real
        3.Cálculo: "Q[e][a] = Q[e][a] + constante de aprendizado*(recompensa real - Q[e][a])"
        4.Caso o jogo não tenha sido finalizado com o movimento, a matriz deve ser atualizada
        5.Deve estimar/prever a próxima melhor Q[estado][ação] e deve ser condizente com a recompensa
        6.Cálculo: "Q[e][a] = Q[e][a] + constante de aprendizado*(maxQ'[proximo e][a']-Q[e][a])"
        7.Equação Final: "Q[e][a] = Q[e][a] + constante de aprendizado*(recompensa(do fim do jogo) + 
        constante de aprendizado alternativa*max(Q[proximo e][a']) - Q[e][a])"
        8.Em python: Q[s][a] += a*(r+g*max([Q[s2][a2] for a in Q[s]])) - Q[s][a]
    '''

    # Alteração: adicionado gamma = 0.9
    # O gamma foi adicionado para conseguir completar o cálculo de previsão do valor Q
    # O valor de 0.9 no gamma foi decidido pois entre 0.1 e 1 ele foi o valor com
    # melhores resultados no desempenho da IA
    def __init__(self, alpha = 0.5, epsilon = 0.1, gamma = 0.9):

        '''
            Initialize AI with an empty Q-learning dictionary,
            an alpha rate and an epsilon rate.

            The Q-learning dictionary maps '(state, action)
            pairs to a Q-value.
                - 'state' is a tuple of remaining piles, e.g. [1, 1, 4, 4]
                - 'action' is a tuple '(i, j)' for an action
        '''
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        # Alteração: adicionado self.gamma = gamma
        # Mesmo motivo de cima
        self.gamma = gamma

    def update_model(self, old_state, action, new_state, reward):

        '''
            Update Q-learning model, given and old state, an action taken
            in that state, a new resulting state, and the reward received
            from taking that action.
        '''

        old_q = self.get_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_value(old_state, action, old_q, reward, best_future)

    def get_value(self, state, action):

        '''
            Return the Q-value for the state 'state' and the action 'action'.
            If no Q-value exists yet in 'self.q', return 0.
        '''
        
        # Raciocinio do código para eu não me perder: self.q (dicionário logo: []"chave": valor])
        # (tuple(state), action) pega o estado atual + a ação e cria uma tupla (s, a)
        # 0 para caso não exista o valor, funciona como um valor padrão porque estava crashando tudo
        value = self.q.get((tuple(state), action), 0)
        return value

    def update_value(self, old_state, action, old_q, reward, future_rewards):

        '''
            Update the Q-value for the state 'state' and the action 'action'
            given the previous Q-value 'old_q', a current reward 'reward',
            and an estimate of future rewards 'future_rewards'.
        '''
        # Prevê um valor da tupla baseado no cálculo que eu escrevi lá em cima:
        # Q[s][a] += alpha*(recompensa+gamma*max([Q[s2][a2] for a in Q[s]])) - Q[s][a]
        # Porém no código max([Q[s2][a2] for a in Q[s]]) pode ser substituido por future_rewards

        new_value = old_q + (self.alpha*((reward + self.gamma*future_rewards)-old_q))

        # OBSERVAÇÃO: Depois que eu troquei self.q[(tuple(old_state), action)] por self.q[tuple(old_state), action]
        # De alguma forma minha IA deixo de ser burra, não sei porque só sei que funcionou
        self.q[tuple(old_state), action] = new_value

    def best_future_reward(self, state):

        '''
            Given a state 'state', consider all possible '(state, action)'
            pairs available in that state and return the maximum of all
            of their Q-values.

            Use 0 as the Q-value if a '(state, action)' pair has no
            Q-value in 'self.q'. If there are no available actions
            in 'state', return 0.
        '''
        game = Nim()

        available_actions = game.available_actions(state)
        if not available_actions:
            return 0

        # OBSERVAÇÃO: O uso de -1e6 é pois os valores atualizados de Q eram alterados
        # para NaN quando eu utilizava valores de infinito no cálculo
        best_q_value = -1e6
        for action in available_actions:
            q_value = self.get_value(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
        return best_q_value

    def choose_action(self, state, epsilon = True):

        '''
            Given a state 'state', return a action '(i, j)' to take.
            
            If 'epsilon' is 'False', then return the best action
            avaiable in the state (the one with the highest Q-value, 
            using 0 for pairs that have no Q-values).

            If 'epsilon' is 'True', then with probability 'self.epsilon'
            chose a random available action, otherwise chose the best
            action available.

            If multiple actions have the same Q-value, any of those
            options is an acceptable return value.
        '''

        # OBSERVAÇÃO: O uso de -1e6 é pois os valores atualizados de Q eram alterados
        # para NaN quando eu utilizava valores de infinito no cálculo
        best_q_value = -1e6
        best_actions = []
        game = Nim()

        # A Conversão para list é para que o random.choice possa escolher alguma ação
        # Do contrário da o erro 'set' object is not subscriptable
        actions = list(game.available_actions(state))

        if epsilon:

            if random.random() < self.epsilon:
                return random.choice(actions)

            for action in actions:
                q_value = self.get_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_actions = [action]
                elif q_value == best_q_value:
                    best_actions.append(action)

            return random.choice(best_actions)
        else:
            best_action = None
            for action in actions:
                q_value = self.get_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            return best_action


def train(player, n_episodes):

    for episode in range(n_episodes):

        print(f'Playing training game {episode + 1}')

        game = Nim()

        # keep track of last move made either player
        last = {0 : {'state' : None, 'action' : None}, 1 : {'state' : None, 'action' : None}}

        while True:

            # keep track of current state and action
            state, action = game.piles.copy(), player.choose_action(game.piles)

            # keep track of last state and action
            last[game.player]['state'], last[game.player]['action'] = state, action

            # make move
            game.move(action)
            new_state = game.piles.copy()

            # when game is over, update Q values with rewards
            if game.winner is not None:
                player.update_model(state, action, new_state, -1)
                player.update_model(last[game.player]['state'], last[game.player]['action'], new_state, 1)
                break
            # if the game is continuing, no rewards yet
            else:
                if last[game.player]['state'] is not None:
                    player.update_model(last[game.player]['state'], last[game.player]['action'], new_state, 0)

    # return the trained player
    return player


def play(ai, human = None):

    # if no player order set, chose human's order randomly
    if human is None:
        human = 0 if random.uniform(0, 1) < 0.5 else 1
        
    # create new game
    game = Nim()

    while True:

        # print contents of piles
        for i, pile in enumerate(game.piles):
            print(f"Pile {i} : {pile}")

        # compute avaiable actions
        # Alteração: Nim.available... -> game.available...
        # Isso foi feito pois estava dando erro no atributo self do método available_actions
        # Esse erro era pois ele stava sendo chamado em uma classe não instânciada
        available_actions = game.available_actions(game.piles)

        # let human make a move
        if game.player == human:
            print('Your turn')
            while True:
                pile = int(input('Choose a pile: '))
                count = int(input('Choose a count: '))
                if (pile, count) in available_actions:
                    break
                print('Invalid move, try again')
        # have AI make a move
        else:
            print('AI turn')
            pile, count = ai.choose_action(game.piles, epsilon = False)
            print(f'AI chose to take {count} from pile {pile}.')
        
        # make move
        game.move((pile, count))
        
        # check for winner
        if game.winner is not None:
            print('GAME OVER')
            winner = 'Human' if game.winner == human else 'AI'
            print(f'Winner is {winner}')
            # Alteração: break adicionado para sair do while True depois do Game Over
            # Input adicionado para ter tempo de ver o resultado antes de passar para o
            # próximo jogo
            input("Aperte |ENTER| para continuar")
            break

def play_against_eachother(sarsa, qlearning):    
    # create new game
    qlearning_turn = 0
    if random.random() < 0.5:
        qlearning_turn = 1

    game = Nim()

    while True:

        # let human make a move
        if game.player == qlearning_turn:
            pile, count = qlearning.choose_action(game.piles, epsilon = False)
        # have AI make a move
        else:
            pile, count = sarsa.choose_action(game.piles, epsilon = False)
        
        # make move
        game.move((pile, count))
        
        # check for winner
        if game.winner is not None:
            winner = 'QLearning' if game.winner == qlearning_turn else 'SARSA'
            return winner