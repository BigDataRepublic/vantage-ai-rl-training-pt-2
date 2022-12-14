import pygame
import random
import settings as s
import glob

class Player(pygame.sprite.Sprite):
    def __init__(self, size=s.PLAYER_SIZE, color=s.ORANGE):
        super(Player, self).__init__()
        
        if s.SPRITES:
            self.surf = pygame.image.load("assets/images/car.png")
            self.surf = pygame.transform.scale(self.surf, size)
        else:
            self.surf = pygame.Surface(size)
            self.surf.fill(color)

        self.rect = self.surf.get_rect()
        self.size = size
        self.s_x = (s.WINDOW_WIDTH-size[0])/2
        self.s_y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.C_x = 0.99
        self.C_y = 0.999
        self.C_f = 0.9
        self.C_r = 0.2
        self.crash = 0.25
        self.left_border = 0
        self.right_border = s.WINDOW_WIDTH
        self.rect.bottom = s.WINDOW_HEIGHT-s.PLAYER_HEIGHT
        
        self.max_speed = 0.0

    def update(self, u_x=0, u_y=0):
        wall_collision = False
        u_x = max(-1, min(u_x, 1))
        u_y = max(-1, min(u_y, 1))
        
        a_x = u_x * 0.5
        a_y = u_y * 0.1
        
        self.v_x = self.C_x * self.v_x + a_x
        self.v_y = self.C_y * self.v_y + a_y 

        # Keep player on the screen
        if self.s_x < self.left_border:
            self.s_x = self.left_border
            self.v_x =  -self.C_r*self.v_x
            self.v_y = self.C_f*self.v_y
            wall_collision = True
        elif self.s_x > self.right_border - self.size[0]:
            self.s_x = self.right_border - self.size[0]
            self.v_x =  -self.C_r*self.v_x
            self.v_y = self.C_f*self.v_y
            wall_collision = True
        
        # Static friction 
        tres = 1e-3
        if abs(self.v_x)<tres:
            self.v_x = 0
        if abs(self.v_y)<tres:
            self.v_y = 0

        self.s_x += self.v_x
        self.s_y += self.v_y
        self.rect.left = self.s_x
        
        if self.v_y > self.max_speed:
            self.max_speed = self.v_y
        
        if s.DEBUG:
            print(f"player velocity: {(self.v_x, self.v_y)}")
            print(f"player distance: {self.s_y}")

        return wall_collision

    def penalize(self):
        self.v_y = self.v_y * self.crash
            

class Enemy(pygame.sprite.Sprite):
    def __init__(self, s_x, s_y, size=s.ENEMY_SIZE, color=s.RED):
        super(Enemy, self).__init__()
        
        if s.SPRITES:
            images = glob.glob("assets/images/enemies/*")
            self.random_image = random.choice(images)
            self.surf = pygame.image.load(self.random_image)
            self.surf = pygame.transform.scale(self.surf, size)
        else:
            self.surf = pygame.Surface(size)
            self.surf.fill(color)
            
        self.rect = self.surf.get_rect()
        self.bottom_border = s.WINDOW_HEIGHT

        self.rect.left = s_x
        self.s_x = s_x
        self.s_y = s_y
        

    def update(self, s_y):
        
        self.rect.bottom = self.bottom_border - s.PLAYER_HEIGHT - (self.s_y - s_y)

        if self.rect.bottom > self.bottom_border + s.ENEMY_SIZE[1]:
            self.kill()
            
class RoadMarker(pygame.sprite.Sprite):
    def __init__(self, s_x, s_y, size=(10,70), color=s.WHITE):
        super(RoadMarker, self).__init__()
        self.surf = pygame.Surface(size)
        self.surf.fill(color)
        # self.surf.set_alpha(100)
        self.rect = self.surf.get_rect()
        self.bottom_border = s.WINDOW_HEIGHT

        self.rect.left = s_x
        self.s_x = s_x
        self.s_y = s_y

    def update(self, s_y):
        self.rect.bottom = self.bottom_border - 50 - (self.s_y - s_y)

        if self.rect.bottom > self.bottom_border + 1000:
            self.kill()


class Finish(pygame.sprite.Sprite):
    def __init__(self, s_x=0, s_y=s.TRACK_LENGTH, size=(s.WINDOW_WIDTH,200), color=s.BLACK):
        super(Finish, self).__init__()
        if s.SPRITES:
            self.surf = pygame.image.load("assets/images/finish.png")
            self.surf = pygame.transform.scale(self.surf, size)
        else:
            self.surf = pygame.Surface(size)
            self.surf.fill(color)

        self.rect = self.surf.get_rect()
        self.bottom_border = s.WINDOW_HEIGHT

        self.rect.left = s_x
        self.s_x = s_x
        self.s_y = s_y


    def update(self, s_y):
        self.rect.bottom = self.bottom_border - 50 - (self.s_y - s_y)