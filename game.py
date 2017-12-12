import sys

import numpy
import pygame
from pygame.locals import *
from pygame.color import *
import random

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

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
        self.angle = 0

        self.shape = pymunk.Circle(self, radius)
        self.shape.color = THECOLORS[player_color]
        self.shape.sensor = True
        self.shape.elasticity = 1.0
        self.shape.collision_type = collision_types["player"]

        space.add(self, self.shape)

    def hurt(self):
        self.score -= 1

    def hit(self):
        self.score += 1

    def act(self, action):
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
            max(self.offset["xmin"], min(self.offset["xmax"], self.position.x + numpy.cos(self.angle) * self.speed)),
            max(self.offset["ymin"], min(self.offset["ymax"], self.position.y + numpy.sin(self.angle) * self.speed))
        )

    def backward(self):
        self.position = (
            max(self.offset["xmin"], min(self.offset["xmax"], self.position.x - numpy.cos(self.angle) * self.speed)),
            max(self.offset["ymin"], min(self.offset["ymax"], self.position.y - numpy.sin(self.angle) * self.speed))
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
            Player(self.space, 15, "red"),
            Player(self.space, 20, "green"),
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

                if player.score == 10:
                    self.game_over = True

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

    def display_frame(self, screen):
        """ Display everything to the screen for the game. """
        screen.fill(pygame.color.THECOLORS["black"])
        font = pygame.font.SysFont("Arial", 16)
        draw_options = pymunk.pygame_util.DrawOptions(screen)

        if self.game_over:
            text = font.render("Game Over, click to restart", True, THECOLORS["red"])
            center_x = (SCREEN_WIDTH // 2) - (text.get_width() // 2)
            center_y = (SCREEN_HEIGHT // 2) - (text.get_height() // 2)
            screen.blit(text, [center_x, center_y])

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
            screen.blit(font.render("Aim with mouse, press to shoot the bullet", 1, THECOLORS["darkgrey"]),
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

    # Create an instance of the Game class
    game = Game()

    # Main game loop
    running = True
    while running:
        # Process events (keystrokes, mouse clicks, etc)
        running = game.process_events()

        # Update object positions, check for collisions
        game.run_logic()

        # Draw the current frame
        game.display_frame(screen)

        # Update players randomly
        for player in game.players:
            action = random.randint(0, 4)
            maybeBullet = player.act(action)
            if maybeBullet is not False:
                game.bullets.append(maybeBullet)

        # Update frame and physics
        fps = 60
        game.update_physics(fps)
        clock.tick(fps)

    # Close window and exit
    pygame.quit()


if __name__ == '__main__':
    sys.exit(main())