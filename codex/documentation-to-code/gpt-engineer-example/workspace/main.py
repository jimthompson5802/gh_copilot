import pygame
from model import Snake, Food, Game
from view import Renderer
from controller import InputHandler

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Snake Game")

    snake = Snake()
    food = Food()
    game = Game(snake, food)
    renderer = Renderer(screen)
    input_handler = InputHandler(snake)

    clock = pygame.time.Clock()

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True
            input_handler.handle_event(event)

        game.update()
        renderer.render(game)

        pygame.display.update()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
