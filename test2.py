import sys

import numpy
import pygame
from pygame.locals import *
from pygame.color import *
import random

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

width, height = 1000, 800
wall_offset = 60
wall_width = 20
collision_types = {
    "player": 1,
    "bullet": 2,
    "wall": 3,
}


class Player(pymunk.Body):

    def __init__(self, space, radius=15, player_color="red", speed=3):
        super().__init__()
        self.offset = {
            "xmin": wall_offset + wall_width + radius,
            "xmax": width - wall_offset - wall_width - radius,
            "ymin": wall_offset + wall_width + radius,
            "ymax": height - wall_offset - wall_width - radius
        }
        self.speed = speed
        self.body_type = pymunk.Body.KINEMATIC
        offset = wall_width + wall_offset
        self.position = (offset + random.randint(0, width - 2 * offset), offset + random.randint(0, height - 2 * offset))
        self.angle = 0

        self.shape = pymunk.Circle(self, radius)
        self.shape.color = THECOLORS[player_color]
        self.shape.sensor = True
        self.shape.elasticity = 1.0
        self.shape.collision_type = collision_types["player"]

        space.add(self, self.shape)

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

    def handle_keys2(self):
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            self.forward()
        elif keys[K_s]:
            self.backward()
        if keys[K_a]:
            self.rotate_left()
        elif keys[K_d]:
            self.rotate_right()

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


class Bullet(pymunk.Body):

    def __init__(self, space, *args, **kwargs):
        super(Bullet, self).__init__(*args, **kwargs)
        self.mass = 1
        self.radius = 2
        self.power = 1000
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.shape = pymunk.Circle(self, self.radius)
        self.shape.friction = .5
        self.shape.collision_type = collision_types["bullet"]

        space.add(self, self.shape)

    def shoot(self, player):
        self.position = player.position + Vec2d(player.shape.radius, 0).rotated(player.angle)
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
    static = [pymunk.Segment(space.static_body, (wall_offset, wall_offset), (wall_offset, height - wall_offset), wall_width)
        , pymunk.Segment(space.static_body, (wall_offset, height - wall_offset), (width - wall_offset, height - wall_offset), wall_width)
        , pymunk.Segment(space.static_body, (width - wall_offset, height - wall_offset), (width - wall_offset, wall_offset), wall_width)
        , pymunk.Segment(space.static_body, (wall_offset, wall_offset), (width - wall_offset, wall_offset), wall_width)
              ]

    player1 = Player(space, 15, "red")
    player2 = Player(space, 20, "green")

    for s in static:
        s.friction = 1.
        s.group = 1
        s.collision_type = collision_types["wall"]
    space.add(static)

    bullets = []

    # Make bricks be removed when hit by ball
    def remove_bullet(arbiter, space, data):
        bullet_shape = arbiter.shapes[0]
        space.remove(bullet_shape, bullet_shape.body)
        bullets.remove(bullet_shape)
        return True

    h = space.add_collision_handler(
        collision_types["bullet"],
        collision_types["wall"]
    )
    h.pre_solve = remove_bullet

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
                bullets.append(bullet.shape)

        player1.handle_keys()
        player2.handle_keys2()

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