import sys
import os
from pandas import DataFrame
import numpy as np
import pygame
import agent as agentFF
import agentLSTM
import os.path
import config
from game import Game
from timer import Timer


def main():
    ### PyGame init
    timer = Timer()
    pygame.init()
    size = [config.SCREEN_WIDTH, config.SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)

    if len(config.players) < config.total_players:
        sys.exit("Not enough player information was provided, " + str(config.total_players) + " players are needed.")

    # Create model learning
    width = config.normalize_coordinate(config.GAME_WIDTH)
    height = config.normalize_coordinate(config.GAME_HEIGHT)
    agents = []
    for x in range(config.total_players):
        name = "model_player_" + str(x) + ".h5"

        if config.use_grid:
            input_size = (config.EXTRA_LAYERS+config.total_players) * width * height
        else:
            input_size = config.DATA_PER_PLAYER * config.total_players

        if config.players[x]["feedforward"]:
            agent = agentFF.SelfLearningAgent(input_size, hidden_size=config.players[x]["hidden_size"])
        else:
            agent = agentLSTM.SelfLearningAgent(input_size, hidden_size=config.players[x]["hidden_size"])

        if os.path.isfile(name):
            print("Model is loaded for agent" + str(x))
            agent.model.load_weights(name)
        agents.append(agent)

    players_won = np.zeros(config.total_players)
    player_won_history = np.zeros((config.total_players, config.epochs))

    for epoch in range(config.epochs):
        print("Running epoch " + str(epoch) + "...")
        # Main game loop
        game = Game(agents)

        for x in range(config.game_length):
            # Update player models
            game.update_models()
            if not game.process_events():
                break

            # Draw the current frame
            if config.display_frame:
                game.display_frame(screen, epoch)

            # Update frame and physics
            game.update_physics(config.fps)

            # Train models on updated data
            game.train_models()
        else:
            best_player = None
            best_score = 0
            for player in game.players:
                if player.score > best_score:
                    best_score = player.score
                    best_player = player.index
            if best_player is not None:
                players_won[best_player - 1] += 1
                print("Player " + str(best_player) + " won epoch " + str(epoch))
            for player in game.players:
                player_won_history[player.index][epoch] = players_won[player.index]
            continue
        break

    print("Total wins per player:")
    print(players_won)

    # Save results to excel file.
    df = DataFrame(data=player_won_history)
    df = df.T
    i = 1
    excel_name = 'game_results'
    while os.path.isfile(excel_name + str(i) + '.xlsx'):
        i += 1
    df.to_excel(excel_name + str(i) + '.xlsx', index=False)

    # Save trained model weights and architecture
    i = 0
    for agent in agents:
        name = "model_player_" + str(i)
        agent.model.save_weights(name + ".h5", overwrite=True)
        i += 1

    # Close window and exit
    pygame.quit()


if __name__ == '__main__':
    sys.exit(main())
