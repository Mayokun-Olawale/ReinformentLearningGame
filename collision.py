import pygame

class HandleCollision(pygame.sprite.Sprite):

    def handle_vertical_collision(player, objects, dy):
        collied_objects = []

        for obj in objects:
            if pygame.sprite.collide_mask(player, obj):
                if dy > 0:
                    player.rect.bottom = obj.rect.top
                    player.landed()
                elif dy < 0:
                    player.rect.top = obj.rect.bottom
                    player.hit_head()
            collied_objects.append(obj)
        return collied_objects