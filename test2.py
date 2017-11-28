"""Showcase of flying bullets that can stick to objects in a somewhat
realistic looking way.
"""
import sys

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util


def create_bullet():
    mass = 1
    moment = pymunk.moment_for_circle(mass, 0, 2)
    bullet_body = pymunk.Body(mass, moment)

    bullet_shape = pymunk.Circle(bullet_body, 2)
    bullet_shape.friction = .5
    bullet_shape.collision_type = 1
    return bullet_body, bullet_shape


def bullet_hit_handler(arbiter, space, data):
    return False


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

    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    static.append(pymunk.Circle(b2, 30))
    b2.position = 300, 400

    for s in static:
        s.friction = 1.
        s.group = 1
    space.add(static)

    # "Cannon" that can fire bullets
    player_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    player_shape = pymunk.Circle(player_body, 25)
    player_shape.sensor = True
    player_shape.color = (255, 50, 50)
    player_body.position = 100, 100
    space.add(player_shape)

    bullet_body, bullet_shape = create_bullet()
    space.add(bullet_shape)

    bullets = []
    handler = space.add_collision_handler(0, 1)
    handler.data["bullets"] = bullets
    handler.pre_solve = space.remove(bullet_shape)

    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                                    event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
                running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(screen, "bullets.png")
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                power = 1000
                impulse = power * Vec2d(1, 0)
                impulse.rotate(bullet_body.angle)

                bullet_body.apply_impulse_at_world_point(impulse, bullet_body.position)

                space.add(bullet_body)
                bullets.append(bullet_body)

                bullet_body, bullet_shape = create_bullet()
                space.add(bullet_shape)

        keys = pygame.key.get_pressed()

        speed = 2.5
        if (keys[K_UP]):
            player_body.position += Vec2d(0, 1) * speed
        if (keys[K_DOWN]):
            player_body.position += Vec2d(0, -1) * speed
        if (keys[K_LEFT]):
            player_body.position += Vec2d(-1, 0) * speed
        if (keys[K_RIGHT]):
            player_body.position += Vec2d(1, 0) * speed

        mouse_position = pymunk.pygame_util.from_pygame(Vec2d(pygame.mouse.get_pos()), screen)
        player_body.angle = (mouse_position - player_body.position).angle
        # move the unfired bullet together with the player
        bullet_body.position = player_body.position + Vec2d(player_shape.radius + 40, 0).rotated(player_body.angle)
        bullet_body.angle = player_body.angle

        for bullet in bullets:
            drag_constant = 0.0002

            pointing_direction = Vec2d(1, 0).rotated(bullet.angle)
            flight_direction = Vec2d(bullet.velocity)
            flight_speed = flight_direction.normalize_return_length()
            dot = flight_direction.dot(pointing_direction)
            # (1-abs(dot)) can be replaced with (1-dot) to make bullets turn
            # around even when fired straight up. Might not be as accurate, but
            # maybe look better.
            drag_force_magnitude = (1 - abs(dot)) * flight_speed ** 2 * drag_constant * bullet.mass
            bullet_tail_position = Vec2d(-50, 0).rotated(bullet.angle)
            bullet.apply_impulse_at_world_point(drag_force_magnitude * -flight_direction, bullet_tail_position)

            bullet.angular_velocity *= 0.5

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])

        ### Draw stuff
        space.debug_draw(draw_options)
        # draw(screen, space)

        # Info and flip screen
        screen.blit(font.render("fps: " + str(clock.get_fps()), 1, THECOLORS["white"]), (0, 0))
        screen.blit(font.render("Aim with mouse, hold LMB to powerup, release to fire", 1, THECOLORS["darkgrey"]),
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