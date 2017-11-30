import sys

import numpy
import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util


class Player(pymunk.Body):

    def __init__(self, radius=15, player_color=(255, 50, 50), speed=3):
        super().__init__()
        self.speed = speed
        self.body_type = pymunk.Body.KINEMATIC
        self.shape = pymunk.Circle(self, radius)
        self.shape.sensor = True
        self.shape.color = player_color
        self.position = 100, 100
        self.angle = 0

    def show(self, space):
        space.add(self.shape)

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

    def forward(self):
        self.position += Vec2d(numpy.cos(self.angle), numpy.sin(self.angle)) * self.speed

    def backward(self):
        self.position -= Vec2d(numpy.cos(self.angle), numpy.sin(self.angle)) * self.speed

    def rotate_left(self):
        self.angle += (self.speed / 100)

    def rotate_right(self):
        self.angle -= (self.speed / 100)


class Bullet(pymunk.Body):

    def __init__(self, space, *args, **kwargs):
        super(Bullet, self).__init__(*args, **kwargs)
        self.mass = 1
        self.radius = 2
        self.power = 1000
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.shape = pymunk.Circle(self, self.radius)
        self.shape.friction = .5
        self.shape.collision_type = 1

        space.add(self.shape)
        space.add(self)

    def shoot(self, player):
        self.position = player.position + Vec2d(player.shape.radius + 40, 0).rotated(player.angle)
        self.angle = player.angle
        impulse = self.power * Vec2d(1, 0)
        impulse.rotate(self.angle)
        self.apply_impulse_at_world_point(impulse, player.position)

    def update(self):
        drag_constant = 0.0002
        pointing_direction = Vec2d(1, 0).rotated(self.angle)
        flight_direction = Vec2d(1, 1)
        flight_speed = flight_direction.normalize_return_length()
        dot = flight_direction.dot(pointing_direction)
        drag_force_magnitude = (1 - abs(dot)) * flight_speed ** 2 * drag_constant * self.mass
        bullet_tail_position = Vec2d(-50, 0).rotated(self.angle)
        print(drag_force_magnitude)
        self.apply_impulse_at_world_point(drag_force_magnitude * -flight_direction, bullet_tail_position)
        self.angular_velocity *= 0.5


width, height = 690, 600


def main():
    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)

    ### Physics stuff
    space = pymunk.Space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # walls - the left-top-right walls
    static = [pymunk.Segment(space.static_body, (50, 50), (50, 550), 5)
        , pymunk.Segment(space.static_body, (50, 550), (650, 550), 5)
        , pymunk.Segment(space.static_body, (650, 550), (650, 50), 5)
        , pymunk.Segment(space.static_body, (50, 50), (650, 50), 5)
              ]

    player1 = Player(15)
    player1.show(space)
    player2 = Player(20, (255, 255, 50))
    player2.show(space)

    for s in static:
        s.friction = 1.
        s.group = 1
    space.add(static)

    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
                running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(screen, "bullets.png")
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                bullet = Bullet(space)
                bullet.shoot(player1)

        player1.handle_keys()

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])

        ### Draw stuff
        space.debug_draw(draw_options)
        # draw(screen, space)

        # Info and flip screen
        screen.blit(font.render("fps: " + str(clock.get_fps()), 1, THECOLORS["white"]), (0, 0))
        screen.blit(font.render("Aim with mouse, press to shoot the bullet", 1, THECOLORS["darkgrey"]),
                    (5, height - 35))
        screen.blit(font.render("Press ESC or Q to quit", 1, THECOLORS["darkgrey"]), (5, height - 20))

        pygame.display.flip()

        ### Update physics
        fps = 60
        dt = 1. / fps
        space.step(dt)

        clock.tick(fps)


if __name__ == '__main__':
    sys.exit(main())