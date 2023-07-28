import pygame
from model import Direction

class InputHandler:
    def __init__(self, snake):
        self.snake = snake

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.snake.direction != Direction.DOWN:
                self.snake.direction = Direction.UP
            elif event.key == pygame.K_DOWN and self.snake.direction != Direction.UP:
                self.snake.direction = Direction.DOWN
            elif event.key == pygame.K_LEFT and self.snake.direction != Direction.RIGHT:
                self.snake.direction = Direction.LEFT
            elif event.key == pygame.K_RIGHT and self.snake.direction != Direction.LEFT:
                self.snake.direction = Direction.RIGHT
