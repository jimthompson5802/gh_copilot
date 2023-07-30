from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass
class Snake:
    body: List[Tuple[int, int]]
    direction: Direction

@dataclass
class Food:
    position: Tuple[int, int]

class Game:
    def __init__(self):
        self.snake = Snake([(10, 10), (10, 11), (10, 12)], Direction.RIGHT)
        self.food = Food((5, 5))
        self.score = 0

    def update(self):
        # Update snake position and check for collisions
        # Update score if food is eaten
        pass
