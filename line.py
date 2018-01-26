import numpy as np


class Line:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination

    def destination_in_front(self):
        AB = self.destination.position - self.origin.position
        normA = (np.cos(self.origin.angle), np.sin(self.origin.angle))
        in_front = np.dot(AB, normA) > 0
        return in_front

    def rotated_angle(self, angle):
        angle = angle / 180 * np.pi
        A = self.destination.position - self.origin.position
        B = (np.cos(self.origin.angle + angle), np.sin(self.origin.angle + angle))
        if np.linalg.norm(A) == 0:
            return np.pi
        if not -1 < np.dot(A, B) / np.linalg.norm(A) < 1:
            return np.pi
        return np.arccos(np.dot(A, B) / np.linalg.norm(A))

    def angle_score(self, angle):
        max_value = np.pi
        return round((max_value - self.rotated_angle(angle)) / max_value, 2)

    def distance_from_line(self):
        return self.distance_from_rotated_line(0)

    def distance_from_rotated_line(self, angle):
        angle = angle / 180 * np.pi
        a = np.tan(self.origin.angle + angle)
        b = -1
        c = self.origin.position.y - a * self.origin.position.x
        distance_from_line = np.abs(a * self.destination.position.x + b * self.destination.position.y + c) / np.sqrt(
            a * a + b * b)
        return distance_from_line

    def distance_score(self, min_distance=500):
        return round(self.distance_rotated_score(0, min_distance), 2)

    def distance_rotated_score(self, angle, min_distance=500):
        distance_to_shooting_line = self.distance_from_rotated_line(angle)
        if distance_to_shooting_line < min_distance:
            return (min_distance - distance_to_shooting_line) / min_distance
        return 0
