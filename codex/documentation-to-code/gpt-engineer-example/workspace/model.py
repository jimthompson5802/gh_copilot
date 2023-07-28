from dataclasses import dataclass
from enum import Enum
import pygame

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass
class Snake:
    position: list
    direction: Direction
    length: int

    def move(self):
        if self.direction == Direction.UP:
            self.position[1] -= 10
        elif self.direction == Direction.DOWN:
            self.position[1] += 10
        elif self.direction == Direction.LEFT:
            self.position[0] -= 10
        elif self.direction == Direction.RIGHT:
            self.position[0] += 10

@dataclass
class Food:
    position: list

    def generate_food(self):
        self.position = [random.randint(0, 79) * 10, random.randint(0, 59) * 10]

class Game:
    def __init__(self, snake, food):
        self.snake = snake
        self.food = food
        self.game_over = False

    def update(self):
        self.snake.move()
        self.check_collision()

    def check_collision(self):
        if self.snake.position == self.food.position:
            self.snake.length += 1
            self.food.generate_food()

        if self.snake.position[0] < 0 or self.snake.position[0] >= 800 or self.snake.position[1] < 0 or self.snake.position[1] >= 600:
            self.game_over = True

        # Check collision with snake's body
        for segment in self.snake.position[1:]:
            if segment == self.snake.position[0]:
                self.game_over = True

