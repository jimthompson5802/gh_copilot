import pygame

class Renderer:
    def __init__(self, screen):
        self.screen = screen

    def render(self, game):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(game.food.position[0], game.food.position[1], 10, 10))

        for segment in game.snake.position:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], 10, 10))
