import pygame
from controller import InputHandler
from model import Game
from view import Renderer

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Snake Game")

    game = Game()
    renderer = Renderer(screen)
    input_handler = InputHandler(game)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            input_handler.handle_event(event)

        game.update()
        renderer.render(game)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
