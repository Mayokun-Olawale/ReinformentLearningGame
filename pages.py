import pygame


def start():
    import game

    background, background_image = game.get_background("Pink.png")
    title_font = pygame.font.Font(None, 96)
    description_font = pygame.font.Font(None, 30)
    button_font = pygame.font.Font(None, 42)
    button_rect = pygame.Rect(390, 520, 270, 80)
    clock = pygame.time.Clock()

    description = "Explore the course, avoid enemies, collect fruit, and reach the trophy."
    running = True
    while running:
        clock.tick(game.FPS)
        mouse_position = pygame.mouse.get_pos()
        button_hovered = button_rect.collidepoint(mouse_position)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    player = game.main(game.window)
                    if player is not None:
                        end(player)
                    return

        for tile in background:
            game.window.blit(background_image, tile)

        title = title_font.render("Welcome", True, (255, 255, 255))
        title_rect = title.get_rect(center=(game.WIDTH // 2, 190))
        game.window.blit(title, title_rect)

        description_surface = description_font.render(description, True, (255, 255, 255))
        description_rect = description_surface.get_rect(center=(game.WIDTH // 2, 330))
        game.window.blit(description_surface, description_rect)

        button_color = (70, 170, 90) if button_hovered else (45, 125, 65)
        pygame.draw.rect(game.window, button_color, button_rect, border_radius=8)
        button_text = button_font.render("Start", True, (255, 255, 255))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        game.window.blit(button_text, button_text_rect)
        pygame.display.update()


def end(player):
    import game

    background, background_image = game.get_background("Pink.png")
    title_font = pygame.font.Font(None, 82)
    stats_font = pygame.font.Font(None, 42)
    button_font = pygame.font.Font(None, 36)
    button_rect = pygame.Rect(390, 570, 270, 70)
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(game.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    running = False

        for tile in background:
            game.window.blit(background_image, tile)

        title_text = {
            "completed": "Course Complete",
            "defeated": "Defeated",
            "fell": "You Fell",
        }.get(player.end_reason, "Game Over")
        title = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title.get_rect(center=(game.WIDTH // 2, 180))
        game.window.blit(title, title_rect)

        lives = stats_font.render(f"Lives remaining: {player.life_count}", True, (255, 255, 255))
        fruit = stats_font.render(f"Fruit collected: {player.fruit_count}", True, (255, 255, 255))
        game.window.blit(lives, lives.get_rect(center=(game.WIDTH // 2, 330)))
        game.window.blit(fruit, fruit.get_rect(center=(game.WIDTH // 2, 400)))

        pygame.draw.rect(game.window, (45, 125, 65), button_rect, border_radius=8)
        button_text = button_font.render("Close", True, (255, 255, 255))
        game.window.blit(button_text, button_text.get_rect(center=button_rect.center))
        pygame.display.update()


if __name__ == "__main__":
    start()
