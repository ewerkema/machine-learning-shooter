import pymunk
from pymunk.vec2d import Vec2d
import config


class Bullet(pymunk.Body):

    def __init__(self, space, player, *args, **kwargs):
        super(Bullet, self).__init__(*args, **kwargs)
        self.mass = 1
        self.radius = 10
        self.power = 1000
        self.moment = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.shape = pymunk.Circle(self, self.radius)
        self.shape.friction = .5
        self.shape.collision_type = config.collision_types["bullet"]
        self.player = player
        space.add(self, self.shape)

        self.position = player.position + Vec2d(player.shape.radius, 0).rotated(player.angle)
        self.angle = player.angle
        impulse = self.power * Vec2d(1, 0)
        impulse.rotate(self.angle)
        self.apply_impulse_at_world_point(impulse, player.position)
