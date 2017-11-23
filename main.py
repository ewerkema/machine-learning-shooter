import pygame,sys
from pygame.locals import *

# screen = pygame.display.set_mode((1024,768))
# screen = pygame.display.set_mode((1024,768), FULLSCEEN)

pygame.init()
DISPLAYSURF = pygame.display.set_mode((400,300))
pygame.display.set_caption('Hello world!')
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()