import pygame
import numpy as np

from pygame.sprite import Sprite, Group


class Mario(Sprite):
    def __init__(self, image, resize_to, ducked_image, ducked_resize_to, X, Y, resolution, horizontal_step, jump_vel,
                 gravity, gene=None):
        super().__init__()
        self.image1 = pygame.transform.scale(pygame.image.load(image), resize_to)
        self.image2 = pygame.transform.scale(pygame.image.load(ducked_image), ducked_resize_to)
        self.image = self.image1
        self.sizeX = pygame.Surface.get_width(self.image)
        self.sizeY = pygame.Surface.get_height(self.image)
        self.rect = self.image.get_rect()
        self.rect.topleft = (X, Y)
        self.init_Y = Y

        self.sizeX1 = pygame.Surface.get_width(self.image1)
        self.sizeY1 = pygame.Surface.get_height(self.image1)
        self.rect1 = self.image1.get_rect()
        self.sizeX2 = pygame.Surface.get_width(self.image2)
        self.sizeY2 = pygame.Surface.get_height(self.image2)
        self.rect2 = self.image2.get_rect()
        self.ydiff = self.sizeY1 - self.sizeY2

        self.resolution = resolution
        self.horizontal_step = horizontal_step
        self.jump_vel = jump_vel
        self.gravity = gravity
        self.is_jumping = False
        self.is_ducked = False
        self.is_going_fw = False
        self.is_going_bw = False
        self.t_jump = 0

    def reset_pos(self):
        x, _ = self.rect.topleft
        self.image = self.image1
        self.sizeX = self.sizeX1
        self.sizeY = self.sizeY1
        self.rect = self.rect1
        self.rect.topleft = (x, self.init_Y)

    def update(self, state):
        if state == "duck" and not self.is_ducked:
            x, y = self.rect.topleft
            self.image = self.image2
            self.sizeX = self.sizeX2
            self.sizeY = self.sizeY2
            self.rect = self.rect2
            self.rect.topleft = (x, y + self.ydiff)
            self.is_ducked = True
        elif state == "fw":
            self.is_going_fw = True
        elif state == "bw":
            self.is_going_bw = True
        elif state == "halt":
            self.is_going_fw = False
            self.is_going_bw = False
        elif state == "jump" and not self.is_ducked:  # can't jump if ducked
            self.is_jumping = True
        elif state == "reset" and self.is_ducked:
            x, y = self.rect.topleft
            self.image = self.image1
            self.sizeX = self.sizeX1
            self.sizeY = self.sizeY1
            self.rect = self.rect1
            self.rect.topleft = (x, y - self.ydiff)
            self.is_ducked = False

        if self.is_going_fw:
            self.rect.x = min(self.resolution[0] - self.sizeX, self.rect.x + self.horizontal_step)
        elif self.is_going_bw:
            self.rect.x = max(0, self.rect.x - self.horizontal_step)

        if self.is_jumping:
            new_y = self.init_Y - (self.jump_vel * self.t_jump - 0.5 * self.gravity * self.t_jump ** 2)
            if new_y > self.init_Y:
                self.rect.y = self.init_Y
                self.t_jump = 0
                self.is_jumping = False
            else:
                self.rect.y = new_y
                self.t_jump += 1


class Mushroom(Sprite):
    def __init__(self, image, resize_to, Y, resolution, released=False):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(image), resize_to)
        self.sizeX = pygame.Surface.get_width(self.image)
        self.sizeY = pygame.Surface.get_height(self.image)
        self.rect = self.image.get_rect()
        self.rect.topleft = (resolution[0], Y)

        self.resolution = resolution
        self.reset_pos()
        self.released = released

    def reset_pos(self):
        self.velociy = np.random.randint(10, 20)
        self.rect.x = self.resolution[0]
        self.released = False

    def update(self):
        if not self.released:
            return
        new_x = self.rect.x - self.velociy
        if new_x > 0:
            self.rect.x = new_x
        else:
            self.reset_pos()


class Fireball(Sprite):
    def __init__(self, image, resize_to, Y, resolution, released=False):
        super().__init__()
        self.image = pygame.transform.scale(pygame.image.load(image), resize_to)
        self.sizeX = pygame.Surface.get_width(self.image)
        self.sizeY = pygame.Surface.get_height(self.image)
        self.rect = self.image.get_rect()
        self.rect.topleft = (resolution[0], Y)

        self.resolution = resolution
        self.reset_pos()
        self.released = released

    def reset_pos(self):
        self.velociy = np.random.randint(20, 30)
        self.rect.x = self.resolution[0]
        self.released = False

    def update(self):
        if not self.released:
            return
        new_x = self.rect.x - self.velociy
        if new_x > 0:
            self.rect.x = new_x
        else:
            self.reset_pos()


class ObstacleGroup(Group):
    def __init__(self, release_count, release_dist):
        super().__init__()
        self.release_count = release_count
        self.release_dist = release_dist

    def update(self, *args):
        sprites = self.sprites()
        farthest = None
        released = 0
        for sprite in sprites:
            if sprite.released:
                released += 1
                if not farthest or farthest.rect.x < sprite.rect.x:
                    farthest = sprite

        if released == 0:
            sprites[np.random.randint(0, len(sprites))].released = True
        elif farthest and farthest.rect.x < self.release_dist:
            while released < self.release_count:
                idx = np.random.randint(0, len(sprites))
                if not sprites[idx].released:
                    sprites[idx].released = True
                    released += 1

        super().update(*args)

    def reset_pos(self):
        for sprite in self.sprites():
            sprite.reset_pos()

class MarioGroup(Group):
    def __init__(self):
        super().__init__()

    def reset_pos(self):
        for sprite in self.sprites():
            sprite.reset_pos()