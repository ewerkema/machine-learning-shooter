import numpy as np
import pygame
from pygame.color import THECOLORS
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import config
from line import Line
from player import Player


class Game(object):
    """ This class represents an instance of the game. If we need to
        reset the game we'd just need to create a new instance of this
        class. """

    def __init__(self, agents):
        """ Constructor. Create all our attributes and initialize
        the game. """

        self.agents = agents
        self.total_players = len(agents)

        self.space = pymunk.Space()
        self.game_over = False

        # Create bullet lists
        self.bullets = []

        # Create the players
        self.players = []

        colors = ["red", "green", "blue", "purple", "yellow"]
        for i in range(self.total_players):
            player = Player(self.space, i, 50, colors[i])
            self.players.append(player)

        # Initialize agents
        self.init_agents()

        # Create walls
        self.create_walls()

        # Create bullet collision handler
        self.init_collision_handlers()

        self.before_state = False
        self.current_state = self.get_data()

    def init_agents(self):
        for agent in self.agents:
            agent.memory.clear()

    def create_walls(self):
        # walls - the left-top-right walls
        top_left = (config.wall_offset, config.wall_offset)
        top_right = (config.SCREEN_WIDTH - config.wall_offset, config.wall_offset)
        bottom_left = (config.wall_offset, config.SCREEN_HEIGHT - config.wall_offset)
        bottom_right = (config.SCREEN_WIDTH - config.wall_offset, config.SCREEN_HEIGHT - config.wall_offset)
        static = [
            pymunk.Segment(self.space.static_body, top_left, top_right, config.wall_width),
            pymunk.Segment(self.space.static_body, top_right, bottom_right, config.wall_width),
            pymunk.Segment(self.space.static_body, bottom_right, bottom_left, config.wall_width),
            pymunk.Segment(self.space.static_body, top_left, bottom_left, config.wall_width),
        ]

        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = config.collision_types["wall"]

        self.space.add(static)

    def init_collision_handlers(self):
        def remove_bullet(arbiter, space, data):
            bullet_shape = arbiter.shapes[0]
            space.remove(bullet_shape, bullet_shape.body)
            for bullet in self.bullets:
                if bullet.shape == bullet_shape:
                    self.bullets.remove(bullet)
            return True

        h = self.space.add_collision_handler(config.collision_types["bullet"], config.collision_types["wall"])
        h.pre_solve = remove_bullet

        def process_bullet_hit(arbiter, space, data):
            bullet_shape = arbiter.shapes[0]
            player_shape = arbiter.shapes[1]

            for bullet in self.bullets:
                if bullet.shape == bullet_shape:
                    bullet.player.hit()

            for player in self.players:
                if player.shape == player_shape:
                    player.hurt()

            return remove_bullet(arbiter, space, data)

        g = self.space.add_collision_handler(config.collision_types["bullet"], config.collision_types["player"])
        g.pre_solve = process_bullet_hit

        def process_players_hit(arbiter, space, data):
            player1_shape = arbiter.shapes[0]
            player2_shape = arbiter.shapes[1]

            for player in self.players:
                if player.shape == player1_shape or player.shape == player2_shape:
                    player.touch_player()

            return True

        k = self.space.add_collision_handler(config.collision_types["player"], config.collision_types["player"])
        k.pre_solve = process_players_hit

    def update_models(self):
        for player in self.players:
            epsilon = 1 if player.random else .1
            action = self.agents[player.index].predict_action(self.current_state, epsilon)
            maybe_bullet = player.act(action)
            if maybe_bullet is not False:
                self.bullets.append(maybe_bullet)

    def train_models(self):
        for player in self.players:
            reward = player.get_reward()
            loss = self.agents[player.index].get_new_state(self.before_state, player.last_action, reward, self.current_state)

    def process_events(self):
        """ Process all of the events. Return a "False" if we need
            to close the window. """

        for event in pygame.event.get():
            if event.type == QUIT or \
                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
                return False
        return True

    def get_data(self):
        if config.use_grid:
            return self.get_grid()
        else:
            return self.get_high_level()

    def get_grid(self):
        width = config.normalize_coordinate(config.GAME_WIDTH)
        height = config.normalize_coordinate(config.GAME_HEIGHT)
        offset = config.wall_width + config.wall_offset
        data = np.zeros((config.EXTRA_LAYERS+self.total_players, width, height))
        for player in self.players:
            index = int(player.index)
            x = config.normalize_coordinate(player.position.x - offset - player.radius)
            y = config.normalize_coordinate(player.position.y - offset - player.radius)
            data[index, x, y] = 1
            shootX = min(width-1, max(0, x + int(round(np.cos(player.angle)))))
            shootY = min(height-1, max(0, y + int(round(np.sin(player.angle)))))
            data[index, shootX, shootY] = 0.5
            for bullet in self.bullets:
                x = config.normalize_coordinate(bullet.position.x - offset)
                y = config.normalize_coordinate(bullet.position.y - offset)
                data[self.total_players, x, y] = 1
        return data.reshape((1, -1))

    def get_high_level(self):
        data = np.zeros(self.total_players * config.DATA_PER_PLAYER)
        i = 0
        for player in self.players:
            for other in self.players:
                if player.index is not other.index:
                    line = Line(player, other)
                    if line.destination_in_front():
                        data[i] = line.distance_score(other.radius) > 0
                    left_score = line.angle_score(10)
                    right_score = line.angle_score(-10)
                    go_left = left_score > right_score
                    data[i + 1] = go_left
                    data[i + 2] = line.angle_score(0)
            for bullet in self.bullets:
                line = Line(bullet, player)
                if line.destination_in_front() and line.distance_from_line() <= player.radius:
                    data[i + 3] = 1
            i += config.DATA_PER_PLAYER
        return data.reshape((1, -1))

    def display_frame(self, screen, epoch):
        """ Display everything to the screen for the game. """
        screen.fill(pygame.color.THECOLORS["black"])
        font = pygame.font.SysFont("Arial", 16)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        if not self.game_over:
            removeLines = []
            if config.debug:
                for player in self.players:
                    for other in self.players:
                        if player.index is not other.index:
                            AB = other.position - player.position
                            normA = (np.cos(player.angle), np.sin(player.angle))
                            in_front = np.dot(AB, normA) > 0
                            if in_front:
                                a = np.tan(player.angle)
                                b = -1
                                c = player.position.y - a * player.position.x
                                angle = 180 * (player.angle % (2 * np.pi)) / np.pi
                                x = 2000 if angle < 90 or angle >= 270 else -2000
                                vision_position = (x, a * x + c)
                                line = pymunk.Segment(self.space.static_body, player.position, vision_position, 1),
                                self.space.add(line)
                                removeLines.append(line)
            # Draw stuff
            self.space.debug_draw(draw_options)
            for line in removeLines:
                self.space.remove(line)

            # Info and flip screen
            scores = ''
            i = 1
            for player in self.players:
                scores += 'Player ' + str(i) + ': ' + str(player.score) + ' ( ' + str(
                    round(player.position.x)) + ', ' + str(round(player.position.y)) + '); action= ' + str(player.last_action)+ '; '
                i += 1
            screen.blit(font.render("Scores= " + scores + " Epoch = " + str(epoch), 1, THECOLORS["white"]), (0, 0))
            screen.blit(font.render("Press ESC or Q to quit", 1, THECOLORS["darkgrey"]), (5, config.SCREEN_HEIGHT - 20))

        pygame.display.flip()

    def update_physics(self, fps):
        self.before_state = self.current_state
        dt = 1. / fps
        self.space.step(dt)
        self.current_state = self.get_data()