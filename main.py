import pygame,sys
from pygame.locals import *

width = 400
height = 300

pygame.init()
GAME = pygame.display.set_mode((width, height))
pygame.display.set_caption('Animation')

FPS = 100 # frames per second setting
fpsClock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
offset = 10
playerx = 10
playery = 10
direction = 'right'

while True:
    GAME.fill(WHITE)

    if direction == 'right':
        playerx += 5
        if playerx == width - offset:
            direction = 'down'
    elif direction == 'down':
        playery += 5
        if playery == height - offset:
            direction = 'left'
    elif direction == 'left':
        playerx -= 5
        if playerx == offset:
            direction = 'up'
    elif direction == 'up':
        playery -= 5
        if playery == offset:
            direction = 'right'

    # GAME.blit(catImg, (playerx, playery))
    pygame.draw.circle(GAME, BLACK, (playerx, playery), 10, 1)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(FPS)