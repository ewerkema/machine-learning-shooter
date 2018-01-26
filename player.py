import random
import numpy as np
import pymunk
from pygame.color import THECOLORS
import config
from bullet import Bullet


class Player(pymunk.Body):

    def __init__(self, space, index, radius=15, player_color="red", speed=3):
        super().__init__()
        self.score = 0
        self.old_score = 0
        self.shot_bullets = 0
        self.hit_bullets = 0
        self.shoot_cooldown = 0
        self.last_action = None
        self.radius = radius
        self.index = index
        self.random = config.players[self.index]['random']
        self.offset = {
            "xmin": config.wall_offset + config.wall_width + radius,
            "xmax": config.SCREEN_WIDTH - config.wall_offset - config.wall_width - radius,
            "ymin": config.wall_offset + config.wall_width + radius,
            "ymax": config.SCREEN_HEIGHT - config.wall_offset - config.wall_width - radius
        }
        self.speed = speed
        self.body_type = pymunk.Body.KINEMATIC
        self.position = (
            self.offset["xmin"] + random.randint(0, self.offset["xmax"] - self.offset["xmin"]),
            self.offset["ymin"] + random.randint(0, self.offset["ymax"] - self.offset["ymin"])
        )
        self.angle = random.randint(0, 360)
        self.shape = pymunk.Circle(self, radius, (0, 0))
        self.shape.color = THECOLORS[player_color]
        self.shape.sensor = True
        self.shape.elasticity = 1.0
        self.shape.collision_type = config.collision_types["player"]

        space.add(self, self.shape)

    def get_reward(self):
        return self.score - self.old_score

    def get_accuracy(self):
        if self.shot_bullets == 0:
            return 0

        return self.hit_bullets / self.shot_bullets * 100

    def hurt(self):
        self.score -= 1

    def hit(self):
        self.score += 1
        self.hit_bullets += 1

    def touch_player(self):
        return

    def act(self, action):
        self.old_score = self.score
        self.last_action = action
        return self.update_state(action)

    def update_state(self, action):
        self.shoot_cooldown -= 1
        if action == config.actions['forward']:
            self.forward()
        elif action == config.actions['backward']:
            self.backward()
        elif action == config.actions['rotate_left']:
            self.rotate_left()
        elif action == config.actions['rotate_right']:
            self.rotate_right()
        elif action == config.actions['shoot']:
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
        self.angle += self.speed * np.pi / 180

    def rotate_right(self):
        self.angle -= self.speed * np.pi / 180

    def shoot(self):
        if self.shoot_cooldown <= 0:
            self.shot_bullets += 1
            self.shoot_cooldown = 10
            return Bullet(self.space, self)
        return False
