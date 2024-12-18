from nim import SARSA, QLearning, train, play, play_against_eachother


# Alteração: dessa vez QLearning e Sarsa estão sendo instanciados, 
# já que no modo anterior não estava passando pelo método __init__ da classe

ai = SARSA()
play(train(player = ai, n_episodes = 1000))

ai = QLearning()
play(train(player = ai, n_episodes = 1000))