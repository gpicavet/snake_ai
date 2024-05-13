# Example file showing a circle moving on screen
import numpy as np
import pygame
import torch
from pygame import Rect

from Game import Game, Direction
from model import LinearQNet

BLOCK_SIZE = 24
BLOCK_GAP = 4
frame_time = 0.01

thegame = Game(20, 20)
thegame.start()

dirMap = {1: Direction.LEFT, 2: Direction.RIGHT, 0: Direction.UP}

model = LinearQNet(len(thegame.get_state()), 256, len(dirMap))
model.load("model.tch")

# pygame setup
pygame.init()
screen = pygame.display.set_mode(((thegame.w + 2) * BLOCK_SIZE + (thegame.w + 1) * BLOCK_GAP,
                                  (thegame.h + 2) * BLOCK_SIZE + (thegame.h + 1) * BLOCK_GAP))
clock = pygame.time.Clock()


def to_rect(pos: (int, int)):
    return Rect((pos[0]+1) * (BLOCK_SIZE + BLOCK_GAP), (pos[1]+1) * (BLOCK_SIZE + BLOCK_GAP), BLOCK_SIZE, BLOCK_SIZE)


def draw(game: Game):
    #walls
    for i in range(game.w + 2):
        pygame.Surface.fill(screen, "gray", to_rect([i-1,-1]))
        pygame.Surface.fill(screen, "gray", to_rect([i-1, game.h]))
    for i in range(game.h):
        pygame.Surface.fill(screen, "gray", to_rect([-1, i]))
        pygame.Surface.fill(screen, "gray", to_rect([game.h, i]))
    #apple
    pygame.Surface.fill(screen, "red", to_rect(game.apple))
    #snake
    pygame.Surface.fill(screen, "green", to_rect(game.snake.body[0]))
    for p in game.snake.body[1:]:
        pygame.Surface.fill(screen, "blue", to_rect(p))


dt = 0
time = 0
running = True
done = False
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    if keys[pygame.K_UP]:
        thegame.dir_human(Direction.UP)
    if keys[pygame.K_DOWN]:
        thegame.dir_human(Direction.DOWN)
    if keys[pygame.K_LEFT]:
        thegame.dir_human(Direction.LEFT)
    if keys[pygame.K_RIGHT]:
        thegame.dir_human(Direction.RIGHT)


    if time>frame_time and not done:
        print(thegame.snake.age)
        print(thegame.snake.body[0], thegame.snake.dir)
        print(thegame.get_state())
        state = np.array(thegame.get_state(), dtype=float)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = model(state0)
        print(prediction)
        move = torch.argmax(prediction).item()
        dir = Direction.LEFT if move==1 else (Direction.RIGHT if move==2 else Direction.UP)
        thegame.dir(dir)

        reward, done, score = thegame.step()
        if done:
            print("game over")
        time=0

    screen.fill("black")
    draw(thegame)

    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000
    time += dt

print("Score: "+str(thegame.score()))

pygame.quit()
