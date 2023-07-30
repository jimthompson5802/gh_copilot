import pygame
from model import Direction

class InputHandler:
    def __init__(self, game):
        self.game = game

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.game.snake.direction = Direction.UP
            elif event.key == pygame.K_DOWN:
                self.game.snake.direction = Direction.DOWN
            elif event.key == pygame.K_LEFT:
                self.game.snake.direction = Direction.LEFT
            elif event.key == pygame.K_RIGHT:
                self.game.snake.direction = Direction.RIGHT
