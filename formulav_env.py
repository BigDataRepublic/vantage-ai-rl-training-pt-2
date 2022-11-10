import pygame
from entities import Player, Enemy, RoadMarker, Finish
import settings as s
import random
import gym
from pygame.locals import ( K_UP,
                            K_DOWN, 
                            K_LEFT,
                            K_RIGHT)
import time 
from utils import try_quit 


class FormulaV(gym.Env):
    """Provides game flow."""

    def __init__(self, seed=s.SEED, human=False):
        super(FormulaV, self).__init__()

        pygame.init()
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        self.screen          = pygame.display.set_mode((s.SCREEN_WIDTH, s.SCREEN_HEIGHT))
        pygame.display.set_caption("Formula V")        
        self.clock           = pygame.time.Clock()
        
        self.seed            = seed
        self.window          = pygame.Surface((s.WINDOW_WIDTH, s.WINDOW_HEIGHT))

        self.player          = Player()
        self.finish_line     = Finish()
        
        self.enemies         = pygame.sprite.Group()
        self.all_sprites     = pygame.sprite.Group()
        self.roadmarkers     = pygame.sprite.Group()
        
        self.running         = True
        self.paused          = False
        self.ticks           = 0
        self.crashes         = 0
        self.obs_index       = 0
        self.marker_index    = 0

        self.done = False 
        self.cumulativereward = 0
        self.reward = 0
        self.action_space = 3
        self.state_space = 11
        self.total, self.best, self.ngames = 0, None, 0
        self.human = human
        self.direction = 0
        
        self.all_sprites.add(self.finish_line)
        self.all_sprites.add(self.player)
        self.obstacles_x, self.obstacles_y = self._generate_obstacle_coords(self.seed)
        self.road_marker_x, self.road_marker_y = self._generate_roadmarker_coords()


    def update(self, action):
        self.reward = 0
        self._add_road_markers()
        self.roadmarkers.update(self.player.s_y)
        
        self._add_enemies()
        
        if self.human:
            u_x, u_y = self._get_control_input(action)
        
        else:
            if action == 0:
                u_x, u_y = -1, 1
                self.direction = -1
            if action == 1:
                u_x, u_y = 1, 1
                self.direction = 1
            if action == 2:
                u_x, u_y = 0, 1

        wall_collision = self.player.update(u_x, u_y)
        self.enemies.update(self.player.s_y)
                
        self.finish_line.update(self.player.s_y)

        collision = self._handle_collisions()
        if collision:
            self.reward -= 1
        if wall_collision:
            self.reward -= 10
        if not collision and not wall_collision:
            self.reward += int(self.player.v_y)

        self._check_finished()
        
        self.ticks += 1

        if s.DEBUG:
            print(f"fps: {self.clock.get_fps()}")
        # print(self.reward)
    
    
    def render(self):
        red = max(0,min(255, self.player.v_y*5))
        self.window.fill((red, 255, 255-red))
    
        for entity in self.all_sprites:
            self.window.blit(entity.surf, entity.rect)
            
        self.window.blit(self.player.surf, self.player.rect)
        
        self.screen.fill(s.BLACK)
        self.screen.blit(self.window, ((s.SCREEN_WIDTH-s.WINDOW_WIDTH)/2, (s.SCREEN_HEIGHT-s.WINDOW_HEIGHT)/2))
        
        font = pygame.font.SysFont('Arial', 16)
        fps = font.render(f"FPS: {round(self.clock.get_fps(),2)}", True, (255,255,255))
        speed = font.render(f"Speed: {round(self.player.v_y,1)}",True,(255,255,255))
        ticks = font.render(f"Ticks: {self.ticks}",True,(255,255,255))
        distance_left = font.render(f"Distance left: {int(s.TRACK_LENGTH-self.player.s_y)}",True,(255,255,255))
        cumulative_reward = font.render(f"Total return: {self.cumulativereward}", True, (255,255,255))
        self.screen.blit(fps,(820,20))
        self.screen.blit(speed, (820, 60))
        self.screen.blit(ticks, (820, 100))
        self.screen.blit(distance_left,(820,140))
        self.screen.blit(cumulative_reward,(820,180))
        
        pygame.display.update()
        
        self.clock.tick(s.FPS)
        
    
    def wait(self):
        while self.paused: 
            pressed_keys = pygame.key.get_pressed()

            if pressed_keys[K_UP]:
                self.paused = False
    
    def _add_enemies(self):
        while (self.obs_index < s.NUM_OBSTACLES) and (self.player.s_y + s.HORIZON > self.obstacles_y[self.obs_index]):

            new_enemy = Enemy(s_x = self.obstacles_x[self.obs_index], s_y = self.obstacles_y[self.obs_index])
            self.enemies.add(new_enemy)
            self.all_sprites.add(new_enemy)
            
            self.obs_index += 1
            
    def _add_road_markers(self):
        while (self.marker_index < s.TRACK_LENGTH) and (self.player.s_y + s.HORIZON > self.road_marker_y[self.marker_index]):

            new_marker = RoadMarker(s_x=self.road_marker_x[self.marker_index], s_y=self.road_marker_y[self.marker_index])
            self.roadmarkers.add(new_marker)
            self.all_sprites.add(new_marker)

            self.marker_index += 1


    def _get_control_input(self, action):
        (u_x, u_y) = action
        
        pressed_keys = pygame.key.get_pressed()
        if s.RENDER and any(pressed_keys):
            
            u_x = u_y = 0
            
            if pressed_keys[K_LEFT]:
                u_x = -1
            if pressed_keys[K_RIGHT]:
                u_x = 1

        return u_x, u_y
    
    
    def _handle_collisions(self):
        collided = pygame.sprite.spritecollideany(self.player, self.enemies)

        if collided:
            self.player.penalize()
            collided.kill()
            self.crashes += 1
            # move_up_sound.stop()
            # move_down_sound.stop()
            # collision_sound.play()   
            return True 
        else:
            return False
              
            
    def _check_finished(self):
        if self.player.s_y > s.TRACK_LENGTH:
            self.running = False
            self.done = True
            

    def _generate_obstacle_coords(self, seed=0):
        if s.NUM_OBSTACLES > 0:
        
            random.seed(seed)
            obstacles_y = [0]
            for i in range(s.NUM_OBSTACLES):
                randint = random.randint(100,200)
                obstacles_y.append(obstacles_y[i] + randint)
    
            obstacles_y = [x/obstacles_y[-1]*(s.TRACK_LENGTH-s.SPAWN_AREA)+s.SPAWN_AREA for x in obstacles_y][:-1]
            obstacles_x = [random.randint(0.5*s.ENEMY_SIZE[0], s.WINDOW_WIDTH - 1.5*s.ENEMY_SIZE[0]) for x in range(len(obstacles_y))]
    
            return obstacles_x, obstacles_y


    def _generate_roadmarker_coords(self):
        marker_y = []
        marker_x = []
        for i in range(int(s.TRACK_LENGTH)):
            marker_y.append(i*125)
            marker_x.append(s.WINDOW_WIDTH*0.25)

            marker_y.append(i*125)
            marker_x.append(s.WINDOW_WIDTH*0.50)

            marker_y.append(i*125)
            marker_x.append(s.WINDOW_WIDTH*0.75)
        return marker_x, marker_y

    
    def step(self, action):
        self.update(action)
        state = self.get_state()
        self.render()
        self.cumulativereward += self.reward
        return state, self.reward, self.done, {}

    
    def get_state(self):
        player = {}
        player["left_wall"] = self.player.s_x / s.WINDOW_WIDTH 
        player["right_wall"] = (s.WINDOW_WIDTH - self.player.s_x - s.PLAYER_SIZE[0]) / s.WINDOW_WIDTH 
        player["distance_traveled"] = self.player.s_y
        player["velocity_x"] = self.player.v_x / 30
        player["velocity_y"] = self.player.v_y / 30
        
        obstacles = []
        for enemy in self.enemies.sprites():
            obstacles.append((enemy.s_x - self.player.s_x)/s.WINDOW_WIDTH)
            obstacles.append((enemy.s_y - self.player.s_y - s.PLAYER_SIZE[1])/s.WINDOW_HEIGHT)

        state = [self.direction, player['velocity_x'], player['velocity_y'], player['left_wall'], player['right_wall']]
        state.extend(obstacles[:6])
        if len(state) < 11:
            state.extend([0,0,0,0,0,0])
        print(state)
        return state[:11]

    
    def reset(self):
        if self.human:
            time.sleep(1)
        self.clock           = pygame.time.Clock()
        self.player          = Player()
        self.finish_line     = Finish()
        
        self.enemies         = pygame.sprite.Group()
        self.all_sprites     = pygame.sprite.Group()
        self.roadmarkers     = pygame.sprite.Group()
        
        self.running         = True
        self.paused          = False
        self.ticks           = 0
        self.crashes         = 0
        self.obs_index       = 0
        self.marker_index    = 0

        self.done = False 
        self.cumulativereward = 0
        self.reward = 0
        self.total = 1
        self.ngames += 1
        
        self.all_sprites.add(self.finish_line)
        self.all_sprites.add(self.player)
        self.obstacles_x, self.obstacles_y = self._generate_obstacle_coords(self.seed)
        self.road_marker_x, self.road_marker_y = self._generate_roadmarker_coords()


if __name__ == '__main__':
    human = True 
    
    env = FormulaV(human=human)

    start_time = time.time()
    while env.running:
        
        for event in pygame.event.get():
            try_quit(event)

        action = (0, 1)
        env.update(action)

        if s.RENDER:
            env.render()
            
    end_time = time.time() - start_time   

    print(f"Game completed in {end_time} seconds.")
    print(f"reached the finish in {env.ticks} ticks!")
    print(f"Number of crashes: {env.crashes}")
    print(f"Max speed reached: {env.player.max_speed} pixels per second!")

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.display.quit()
    pygame.quit()
