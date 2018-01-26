import numpy as np
import sys
import pygame
from pandas import DataFrame
import config
from game import Game
import agentFF
import agentLSTM
import os.path


class World(object):
    def __init__(self):
        pygame.init()
        self.size = [config.SCREEN_WIDTH, config.SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(self.size)

        if len(config.players) < config.total_players:
            sys.exit(
                "Not enough player information was provided, " + str(config.total_players) + " players are needed."
            )

        self.agents = []
        self.players_won = np.zeros(config.total_players)
        self.player_won_history = np.zeros((config.total_players, config.epochs))

        self.init_models()

    def init_models(self):
        input_size = Game.get_data_size()
        for index, player in enumerate(config.players):
            name = "model_player_" + str(index) + ".h5"

            if player["feedforward"]:
                agent = agentFF.Agent(input_size, hidden_size=player["hidden_size"])
            else:
                agent = agentLSTM.Agent(input_size, hidden_size=player["hidden_size"])

            if os.path.isfile(name):
                print("Model is loaded for agent" + str(index))
                agent.model.load_weights(name)
            self.agents.append(agent)

    def run_epoch(self, epoch):
        print("Running epoch " + str(epoch) + "...")
        game = Game(self.agents, epoch)

        for x in range(config.game_length):
            if not game.process_events():
                return False

            game.run(self.screen)

        best_player = game.best_player()
        if best_player is not None:
            self.players_won[best_player] += 1
            print("Player " + str(best_player) + " won epoch " + str(epoch))
        for player in game.players:
            self.player_won_history[player.index][epoch] = self.players_won[player.index]

        return True

    def save_results_to_excel(self):
        # Save results to excel file.
        df = DataFrame(data=self.player_won_history)
        df = df.T
        i = 1
        excel_name = 'game_results'
        while os.path.isfile(excel_name + str(i) + '.xlsx'):
            i += 1
        df.to_excel(excel_name + str(i) + '.xlsx', index=False)

    def save_models(self):
        for index, agent in enumerate(self.agents):
            name = "model_player_" + str(index)
            agent.model.save_weights(name + ".h5", overwrite=True)

    def quit(self):
        # Close window and exit
        pygame.quit()
