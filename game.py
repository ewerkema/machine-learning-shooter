import sys

import numpy as np
import pygame
import json
from pygame.locals import *
from pygame.color import *
import random
import agent as learn
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import matplotlib.pyplot as plt
import os.path

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800
wall_offset = 60
wall_width = 20
collision_types = {
    "player": 1,
    "bullet": 2,
    "wall": 3,
}
FORWARD = 0
BACKWARD = 1
ROTATE_LEFT = 2
ROTATE_RIGHT = 3
SHOOT = 4


class Player(pymunk.Body):

    def __init__(self, space, radius=15, player_color="red", speed=3):
        super().__init__()
        self.score = 0
        self.old_score = 0
        self.last_action = None
        self.offset = {
            "xmin": wall_offset + wall_width + radius,
            "xmax": SCREEN_WIDTH - wall_offset - wall_width - radius,
            "ymin": wall_offset + wall_width + radius,
            "ymax": SCREEN_HEIGHT - wall_offset - wall_width - radius
        }
        self.speed = speed
        self.body_type = pymunk.Body.KINEMATIC
        self.position = (
            self.offset["xmin"] + random.randint(0, self.offset["xmax"] - self.offset["xmin"]),
            self.offset["ymin"] + random.randint(0, self.offset["ymax"] - self.offset["ymin"])
        )
        self.angle = random.randint(0, 360)

        self.shape = pymunk.Circle(self, radius)
        self.shape.color = THECOLORS[player_color]
        self.shape.sensor = True
        self.shape.elasticity = 1.0
        self.shape.collision_type = collision_types["player"]

        space.add(self, self.shape)

    def get_reward(self):
        return self.score - self.old_score

    def hurt(self):
        self.score -= 1

    def hit(self):
        self.score += 1

    def act(self, action):
        self.old_score = self.score
        self.last_action = action
        return self.update_state(action)

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            self.forward()
        elif keys[K_DOWN]:
            self.backward()
        if keys[K_LEFT]:
            self.rotate_left()
        elif keys[K_RIGHT]:
            self.rotate_right()

    def update_state(self, action):
        if action == FORWARD:
            self.forward()
        elif action == BACKWARD:
            self.backward()
        elif action == ROTATE_LEFT:
            self.rotate_left()
        elif action == ROTATE_RIGHT:
            self.rotate_right()
        elif action == SHOOT:
            bullet = self.shoot()
            return bullet

        return False

    def forward(self):
        self.position = (
            max(self.offset["xmin"], min(self.offset["xmax"], self.position.x + np.cos(self.angle) * self.speed)),
            max(self.offset["ymin"], min(self.offset["ymax"], self.position.y + np.sin(self.angle) * self.speed))
        )

    def backward(self):
        self.position = (
            max(self.offset["xmin"], min(self.offset["xmax"], self.position.x - np.cos(self.angle) * self.speed)),
            max(self.offset["ymin"], min(self.offset["ymax"], self.position.y - np.sin(self.angle) * self.speed))
        )

    def rotate_left(self):
        self.angle += (self.speed * 2 / 100)

    def rotate_right(self):
        self.angle -= (self.speed * 2 / 100)

    def shoot(self):
        return Bullet(self.space, self)


class Bullet(pymunk.Body):

    def __init__(self, space, player, *args, **kwargs):
        super(Bullet, self).__init__(*args, **kwargs)
        self.mass = 1
        self.radius = 2
        self.power = 1000
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.shape = pymunk.Circle(self, self.radius)
        self.shape.friction = .5
        self.shape.collision_type = collision_types["bullet"]
        self.player = player
        space.add(self, self.shape)

        self.position = player.position + Vec2d(player.shape.radius, 0).rotated(player.angle)
        self.angle = player.angle
        impulse = self.power * Vec2d(1, 0)
        impulse.rotate(self.angle)
        self.apply_impulse_at_world_point(impulse, player.position)


class Game(object):
    """ This class represents an instance of the game. If we need to
        reset the game we'd just need to create a new instance of this
        class. """

    def __init__(self):
        """ Constructor. Create all our attributes and initialize
        the game. """

        self.space = pymunk.Space()
        self.game_over = False

        # Create bullet lists
        self.bullets = []

        # Create the players
        self.players = [
            Player(self.space, 50, "red"),
            Player(self.space, 50, "green"),
        ]

        # Create walls
        self.create_walls()

        # Create bullet collision handler
        self.init_collision_handlers()

    def create_walls(self):
        # walls - the left-top-right walls
        top_left = (wall_offset, wall_offset)
        top_right = (wall_offset, SCREEN_HEIGHT - wall_offset)
        bottom_left = (SCREEN_WIDTH - wall_offset, wall_offset)
        bottom_right = (SCREEN_WIDTH - wall_offset, SCREEN_HEIGHT - wall_offset)

        static = [
            pymunk.Segment(self.space.static_body, top_left, top_right, wall_width),
            pymunk.Segment(self.space.static_body, top_right, bottom_right, wall_width),
            pymunk.Segment(self.space.static_body, bottom_right, bottom_left, wall_width),
            pymunk.Segment(self.space.static_body, top_left, bottom_left, wall_width),
        ]

        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = collision_types["wall"]

        self.space.add(static)

    def init_collision_handlers(self):
        def remove_bullet(arbiter, space, data):
            bullet_shape = arbiter.shapes[0]
            space.remove(bullet_shape, bullet_shape.body)
            for bullet in self.bullets:
                if bullet.shape == bullet_shape:
                    self.bullets.remove(bullet)
            return True

        h = self.space.add_collision_handler(collision_types["bullet"], collision_types["wall"])
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

        g = self.space.add_collision_handler(collision_types["bullet"], collision_types["player"])
        g.pre_solve = process_bullet_hit

    def run_logic(self):
        """
        This method is run each time through the frame. It
        updates positions and checks for collisions.
        """
        if not self.game_over:
            # Handle keys of players
            for player in self.players:
                player.handle_keys()

    def process_events(self):
        """ Process all of the events. Return a "False" if we need
            to close the window. """

        for event in pygame.event.get():
            if event.type == QUIT or \
                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
                return False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if not self.game_over:
                    bullet = self.players[0].shoot()
                    self.bullets.append(bullet)
                else:
                    self.__init__()
        return True

    def get_data(self):
        data = np.zeros(10)
        i = 0
        for player in self.players:
            data[i] = player.position.x
            data[i+1] = player.position.y
            data[i+2] = player.angle
            i += 3
        for bullet in self.bullets:
            if i > 9:
                break

            data[i] = bullet.position.x
            data[i+1] = bullet.position.y
            i += 2
        return data.reshape((1, -1))

    def display_frame(self, screen, q1, q2):
        """ Display everything to the screen for the game. """
        screen.fill(pygame.color.THECOLORS["black"])
        font = pygame.font.SysFont("Arial", 16)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        # if self.game_over:
        #     text = font.render("Game Over, click to restart", True, THECOLORS["red"])
        #     center_x = (SCREEN_WIDTH // 2) - (text.get_width() // 2)
        #     center_y = (SCREEN_HEIGHT // 2) - (text.get_height() // 2)
        #     screen.blit(text, [center_x, center_y])

        if not self.game_over:
            # Draw stuff
            self.space.debug_draw(draw_options)

            # Info and flip screen
            scores = ''
            i = 1
            for player in self.players:
                scores += 'Player ' + str(i) + ': ' + str(player.score) + ' '
                i += 1
            screen.blit(font.render("Scores: " + scores, 1, THECOLORS["white"]), (0, 0))
            screen.blit(font.render("Player 1: " + str(q1[0:5]) + ", Player 2: " + str(q2[0:5]), 1, THECOLORS["darkgrey"]),
                        (5, SCREEN_HEIGHT - 35))
            screen.blit(font.render("Press ESC or Q to quit", 1, THECOLORS["darkgrey"]), (5, SCREEN_HEIGHT - 20))

        pygame.display.flip()

    def update_physics(self, fps):
        dt = 1. / fps
        self.space.step(dt)


def main():
    ### PyGame init
    pygame.init()
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)

    clock = pygame.time.Clock()

    # Create model learning
    total_players = 2
    agents = []
    for x in range(total_players):
        name = "model_player_" + str(x) + ".h5"
        agent = learn.SelfLearningAgent(2, 2)
        if os.path.isfile(name):
            print("Model is loaded for agent" + str(x))
            agent.model.load_weights(name)
        agents.append(agent)

    epochs = 5
    fps = 25
    game_length = fps * 15
    rewards = np.zeros(epochs*game_length)

    for epoch in range(epochs):
        # Main game loop
        game = Game()
        for x in range(game_length):
            # Update object positions, check for collisions
            game.run_logic()
            game.process_events()

            # Update players based on model
            before_data = game.get_data()
            i = 0
            for player in game.players:
                action = agents[i].predict_action(before_data)
                maybe_bullet = player.act(action)
                if maybe_bullet is not False:
                    game.bullets.append(maybe_bullet)
                i += 1

            # Draw the current frame
            game.display_frame(screen, agents[0].model.predict(before_data)[0], agents[1].model.predict(before_data)[0])

            # Update frame and physics
            game.update_physics(fps)
            clock.tick(fps)

            after_data = game.get_data()
            i = 0
            for player in game.players:
                reward = player.get_reward()
                loss = agents[i].get_new_state(before_data, player.last_action, reward, after_data)
                rewards[(epoch * game_length) + x] = loss
                i += 1

    x = range(0, epochs * game_length)
    plt.plot(x, rewards)
    # axes = plt.gca()
    # axes.set_ylim([0, grid_size*3])
    plt.show()

    # Save trained model weights and architecture, this will be used by the visualization code
    i = 0
    for agent in agents:
        name = "model_player_" + str(i)
        agent.model.save_weights(name + ".h5", overwrite=True)
        # with open(name + ".json", "w") as outfile:
        #     json.dump(agent.model.to_json(), outfile)
        i += 1

    # Close window and exit
    pygame.quit()


if __name__ == '__main__':
    sys.exit(main())
