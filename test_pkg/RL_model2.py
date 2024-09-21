# import gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# class MobileRobotEnv(gym.Env):
#     def __init__(self):
#         super(MobileRobotEnv, self).__init__()
        
#         # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
#         self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
#                                                 high=np.array([4, 4, 4, 4]), 
#                                                 dtype=np.float32)
        
#         # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
#         self.action_space = gym.spaces.Discrete(4)
        
#         # 로봇 초기 위치
#         self.robot_pos = np.array([4, 4])
        
#         # 장애물 위치 설정
#         self.obstacles = [np.array([0, 0]), np.array([1, 1]), np.array([3, 0]),
#                           np.array([0, 1]), np.array([1, 1]), np.array([3, 1]),
#                           np.array([0, 2]), np.array([0, 3])]
        
#         # 목표 위치 랜덤 초기화
#         self.goal_pos = self._generate_random_goal()
    
#     def _generate_random_goal(self):
#         while True:
#             goal_pos = np.random.randint(0, 5, size=2)
#             if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles):
#                 return goal_pos
    
#     def reset(self):
#         # 로봇의 위치를 초기화하고 목표 위치도 재설정
#         self.robot_pos = np.array([4, 4])
#         self.goal_pos = self._generate_random_goal()
#         return self._get_obs()
    
#     def step(self, action):
#         # 로봇의 행동 적용
#         if action == 0 and self.robot_pos[1] < 4:  # 위로 이동
#             self.robot_pos[1] += 1
#         elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
#             self.robot_pos[1] -= 1
#         elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
#             self.robot_pos[0] -= 1
#         elif action == 3 and self.robot_pos[0] < 4:  # 오른쪽으로 이동
#             self.robot_pos[0] += 1
        
#         # 보상 계산 및 완료 여부 확인
#         reward = self._calculate_reward()
#         done = np.array_equal(self.robot_pos, self.goal_pos)
        
#         return self._get_obs(), reward, done, {}
    
#     def _get_obs(self):
#         # 로봇의 위치와 목표물의 위치 반환
#         return np.concatenate((self.robot_pos, self.goal_pos))
    
#     def _calculate_reward(self):
#         # 장애물에 충돌하면 큰 패널티
#         for obstacle in self.obstacles:
#             if np.array_equal(self.robot_pos, obstacle):
#                 return -10
        
#         # 목표에 도달하면 큰 보상
#         if np.array_equal(self.robot_pos, self.goal_pos):
#             return 100
        
#         # 목표로부터의 거리 감소에 따른 보상
#         distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
#         reward = -distance_to_goal
        
#         return reward
    
#     def render(self, mode='human'):
#         # 환경을 텍스트로 시각화
#         env_map = np.zeros((5, 5), dtype=str)
#         env_map[:, :] = '.'
#         env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
#         env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
#         for obstacle in self.obstacles:
#             env_map[obstacle[1], obstacle[0]] = 'X'
        
#         print("\n".join([" ".join(row) for row in env_map]))
#         print()

# # 환경 초기화
# env = MobileRobotEnv()
# env = DummyVecEnv([lambda: env])

# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)

# # 학습된 모델로 로봇 테스트
# print("학습 완료 후 테스트:")
# obs = env.reset()
# env.render()  # 초기 상태 렌더링
# for _ in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
#     if dones:
#         break

# model.save("ppo_mobile_robot")




# import gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv



# class MobileRobotEnv(gym.Env):

#     def __init__(self):
#         super(MobileRobotEnv, self).__init__()

        
#         # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
#         self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
#                                                 high=np.array([4, 4, 4, 4]), 
#                                                 dtype=np.float32)

        
#         # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
#         self.action_space = gym.spaces.Discrete(4)

        
#         # 장애물 위치 설정
#         self.obstacles = [np.array([0, 0]), np.array([1, 1]), np.array([3, 0]),
#                           np.array([0, 1]), np.array([1, 1]), np.array([3, 1]),
#                           np.array([0, 2]), np.array([0, 3])]

        

#         # 로봇 시작 위치 및 목표 위치 랜덤 초기화
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()

    
#     def _generate_random_goal(self):

#         while True:
#             goal_pos = np.random.randint(0, 5, size=2)

#             if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles):
#                 return goal_pos

    

#     def _generate_random_start(self):

#         while True:
#             start_pos = np.random.randint(0, 5, size=2)

#             if not any(np.array_equal(start_pos, obs) for obs in self.obstacles):
#                 return start_pos



#     def reset(self):

#         # 로봇의 위치와 목표 위치를 랜덤하게 재설정
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()
#         return self._get_obs()

    

#     def step(self, action):

#         # 로봇의 행동 적용
#         if action == 0 and self.robot_pos[1] < 4:  # 위로 이동
#             self.robot_pos[1] += 1

#         elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
#             self.robot_pos[1] -= 1

#         elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
#             self.robot_pos[0] -= 1

#         elif action == 3 and self.robot_pos[0] < 4:  # 오른쪽으로 이동
#             self.robot_pos[0] += 1

        
#         # 보상 계산 및 완료 여부 확인
#         reward = self._calculate_reward()
#         done = np.array_equal(self.robot_pos, self.goal_pos)
#         return self._get_obs(), reward, done, {}

    
#     def _get_obs(self):
#         # 로봇의 위치와 목표물의 위치 반환
#         return np.concatenate((self.robot_pos, self.goal_pos))

    

#     def _calculate_reward(self):
#         # 장애물에 충돌하면 큰 패널티
#         for obstacle in self.obstacles:
#             if np.array_equal(self.robot_pos, obstacle):
#                 return -10

    
#         # 목표에 도달하면 큰 보상
#         if np.array_equal(self.robot_pos, self.goal_pos):
#             return 100


#         # 목표로부터의 거리 감소에 따른 보상
#         distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
#         reward = -distance_to_goal
#         return reward

    

#     def render(self, mode='human'):
#         # 환경을 텍스트로 시각화
#         env_map = np.zeros((5, 5), dtype=str)
#         env_map[:, :] = '.'
#         env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
#         env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'

#         for obstacle in self.obstacles:
#             env_map[obstacle[1], obstacle[0]] = 'X'

#         print("\n".join([" ".join(row) for row in env_map]))
#         print()



# # 환경 초기화
# env = MobileRobotEnv()
# env = DummyVecEnv([lambda: env])



# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)



# # 학습된 모델로 로봇 테스트
# print("학습 완료 후 테스트:")
# obs = env.reset()
# env.render()  # 초기 상태 렌더링

# for _ in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

#     if dones:
#         break


# #model.save("ppo_mobile_robot")





# import gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# class MobileRobotEnv(gym.Env):
#     def __init__(self, render_mode=None):
#         super(MobileRobotEnv, self).__init__()
        
#         self.render_mode = render_mode
        
#         # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
#         self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
#                                                 high=np.array([4, 4, 4, 4]), 
#                                                 dtype=np.float32)
        
#         # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
#         self.action_space = gym.spaces.Discrete(4)
        
#         # 장애물 위치 설정
#         self.obstacles = [np.array([0, 0]), np.array([1, 1]), np.array([3, 0]),
#                           np.array([0, 1]), np.array([1, 1]), np.array([3, 1]),
#                           np.array([0, 2]), np.array([0, 3])]
        
#         # 로봇 시작 위치 및 목표 위치 랜덤 초기화
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()
    
#     def _generate_random_goal(self):
#         while True:
#             goal_pos = np.random.randint(0, 5, size=2)
#             if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles):
#                 return goal_pos
    
#     def _generate_random_start(self):
#         while True:
#             start_pos = np.random.randint(0, 5, size=2)
#             if not any(np.array_equal(start_pos, obs) for obs in self.obstacles):
#                 return start_pos

#     def reset(self):
#         # 로봇의 위치와 목표 위치를 랜덤하게 재설정
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()
#         return self._get_obs()
    
#     def step(self, action):
#         # 로봇의 행동 적용
#         if action == 0 and self.robot_pos[1] < 4:  # 위로 이동
#             self.robot_pos[1] += 1
#         elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
#             self.robot_pos[1] -= 1
#         elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
#             self.robot_pos[0] -= 1
#         elif action == 3 and self.robot_pos[0] < 4:  # 오른쪽으로 이동
#             self.robot_pos[0] += 1
        
#         # 보상 계산 및 완료 여부 확인
#         reward = self._calculate_reward()
#         done = np.array_equal(self.robot_pos, self.goal_pos)
        
#         if self.render_mode == 'human':
#             self.render()

#         return self._get_obs(), reward, done, {}
    
#     def _get_obs(self):
#         # 로봇의 위치와 목표물의 위치 반환
#         return np.concatenate((self.robot_pos, self.goal_pos))
    
#     def _calculate_reward(self):
#         # 장애물에 충돌하면 큰 패널티
#         for obstacle in self.obstacles:
#             if np.array_equal(self.robot_pos, obstacle):
#                 return -10
        
#         # 목표에 도달하면 큰 보상
#         if np.array_equal(self.robot_pos, self.goal_pos):
#             return 100
        
#         # 목표로부터의 거리 감소에 따른 보상
#         distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
#         reward = -distance_to_goal
        
#         return reward
    
#     def render(self, mode='human'):
#         # 환경을 텍스트로 시각화
#         env_map = np.zeros((5, 5), dtype=str)
#         env_map[:, :] = '.'
#         env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
#         env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
#         for obstacle in self.obstacles:
#             env_map[obstacle[1], obstacle[0]] = 'X'
        
#         print("\n".join([" ".join(row) for row in env_map]))
#         print()

# # 환경 초기화 (render_mode 설정)
# env = MobileRobotEnv(render_mode='human')
# env = DummyVecEnv([lambda: env])

# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)

# # 학습된 모델로 로봇 테스트
# print("학습 완료 후 테스트:")
# obs = env.reset()
# env.render()  # 초기 상태 렌더링
# for _ in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     if dones:
#         break

# #model.save("ppo_mobile_robot")






# import gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# class MobileRobotEnv(gym.Env):
#     def __init__(self, render_mode=None):
#         super(MobileRobotEnv, self).__init__()
        
#         self.render_mode = render_mode
        
#         # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
#         self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
#                                                 high=np.array([4, 4, 4, 4]), 
#                                                 dtype=np.float32)
        
#         # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
#         self.action_space = gym.spaces.Discrete(4)
        
#         # 장애물 위치 설정
#         self.obstacles = [np.array([0, 0]), np.array([1, 1]), np.array([3, 0]),
#                           np.array([0, 1]), np.array([1, 1]), np.array([3, 1]),
#                           np.array([0, 2]), np.array([0, 3])
                          
#                           ]
        
#         # 로봇 시작 위치 및 목표 위치 랜덤 초기화
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()
    
#     def _generate_random_goal(self):
#         while True:
#             goal_pos = np.random.randint(0, 5, size=2)
#             if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles) and not np.array_equal(goal_pos, self.robot_pos):
#                 return goal_pos
    
#     def _generate_random_start(self):
#         while True:
#             start_pos = np.random.randint(0, 5, size=2)
#             if not any(np.array_equal(start_pos, obs) for obs in self.obstacles):
#                 return start_pos

#     def reset(self):
#         # 로봇의 위치와 목표 위치를 랜덤하게 재설정
#         self.robot_pos = self._generate_random_start()
#         self.goal_pos = self._generate_random_goal()
#         return self._get_obs()
    
#     def step(self, action):
#         # 로봇의 행동 적용
#         if action == 0 and self.robot_pos[1] < 4:  # 위로 이동
#             self.robot_pos[1] += 1
#         elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
#             self.robot_pos[1] -= 1
#         elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
#             self.robot_pos[0] -= 1
#         elif action == 3 and self.robot_pos[0] < 4:  # 오른쪽으로 이동
#             self.robot_pos[0] += 1
        
#         # 보상 계산 및 완료 여부 확인
#         reward = self._calculate_reward()
#         done = np.array_equal(self.robot_pos, self.goal_pos)
        
#         if self.render_mode == 'human':
#             self.render()

#         return self._get_obs(), reward, done, {}
    
#     def _get_obs(self):
#         # 로봇의 위치와 목표물의 위치 반환
#         return np.concatenate((self.robot_pos, self.goal_pos))
    
#     def _calculate_reward(self):
#         # 장애물에 충돌하면 큰 패널티
#         for obstacle in self.obstacles:
#             if np.array_equal(self.robot_pos, obstacle):
#                 return -10
        
#         # 목표에 도달하면 큰 보상
#         if np.array_equal(self.robot_pos, self.goal_pos):
#             return 100
        
#         # 목표로부터의 거리 감소에 따른 보상
#         distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
#         reward = -distance_to_goal
        
#         return reward
    
#     def render(self, mode='human'):
#         # 환경을 텍스트로 시각화
#         env_map = np.zeros((5, 5), dtype=str)
#         env_map[:, :] = '.'
#         env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
#         env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
#         for obstacle in self.obstacles:
#             env_map[obstacle[1], obstacle[0]] = 'X'
        
#         print("\n".join([" ".join(row) for row in env_map]))
#         print()

# # 환경 초기화 (render_mode 설정)
# env = MobileRobotEnv(render_mode='human')
# env = DummyVecEnv([lambda: env])

# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)

# # 학습된 모델로 로봇 테스트
# print("학습 완료 후 테스트:")
# obs = env.reset()
# env.render()  # 초기 상태 렌더링
# for _ in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     if dones:
#         break

# #model.save("ppo_mobile_robot")



import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MobileRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MobileRobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.grid_size = 7  # 그리드 크기 설정
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
                                                high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]), 
                                                dtype=np.float32)
        
        # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
        self.action_space = gym.spaces.Discrete(4)
        
        # 장애물 위치 설정 (가장자리 및 추가 장애물 배치)
        self.obstacles = [np.array([0, i]) for i in range(self.grid_size)] + \
                         [np.array([self.grid_size-1, i]) for i in range(self.grid_size)] + \
                         [np.array([i, 0]) for i in range(1, self.grid_size-1)] + \
                         [np.array([i, self.grid_size-1]) for i in range(1, self.grid_size-1)] + \
                         [np.array([1, 1]), np.array([1, 2]), np.array([2, 2]), np.array([1, 3]), 
                          np.array([1, 4]), np.array([4, 1]), np.array([4, 2])]

        # 로봇 시작 위치 및 목표 위치 랜덤 초기화
        self.robot_pos = self._generate_random_start()
        self.goal_pos = self._generate_random_goal()
    
    def _generate_random_goal(self):
        while True:
            goal_pos = np.random.randint(1, self.grid_size-1, size=2)
            if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles) and not np.array_equal(goal_pos, self.robot_pos):
                return goal_pos
    
    def _generate_random_start(self):
        while True:
            start_pos = np.random.randint(1, self.grid_size-1, size=2)
            if not any(np.array_equal(start_pos, obs) for obs in self.obstacles):
                return start_pos

    def reset(self):
        # 로봇의 위치와 목표 위치를 랜덤하게 재설정
        self.robot_pos = self._generate_random_start()
        self.goal_pos = self._generate_random_goal()
        return self._get_obs()
    
    def step(self, action):
        # 로봇의 행동 적용
        if action == 0 and self.robot_pos[1] < self.grid_size-1:  # 위로 이동
            self.robot_pos[1] += 1
        elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
            self.robot_pos[1] -= 1
        elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
            self.robot_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.grid_size-1:  # 오른쪽으로 이동
            self.robot_pos[0] += 1
        
        # 보상 계산 및 완료 여부 확인
        reward = self._calculate_reward()
        done = np.array_equal(self.robot_pos, self.goal_pos)
        
        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        # 로봇의 위치와 목표물의 위치 반환
        return np.concatenate((self.robot_pos, self.goal_pos))
    
    def _calculate_reward(self):
        # 장애물에 충돌하면 큰 패널티
        for obstacle in self.obstacles:
            if np.array_equal(self.robot_pos, obstacle):
                return -10
        
        # 목표에 도달하면 큰 보상
        if np.array_equal(self.robot_pos, self.goal_pos):
            return 100
        
        # 목표로부터의 거리 감소에 따른 보상
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -distance_to_goal
        
        return reward
    
    def render(self, mode='human'):
        # 환경을 텍스트로 시각화
        env_map = np.zeros((self.grid_size, self.grid_size), dtype=str)
        env_map[:, :] = '.'
        env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
        env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
        for obstacle in self.obstacles:
            env_map[obstacle[1], obstacle[0]] = 'X'
        
        print("\n".join([" ".join(row) for row in env_map]))
        print()

# 환경 초기화 (render_mode 설정)
env = MobileRobotEnv(render_mode='human')
env = DummyVecEnv([lambda: env])

# PPO 모델 초기화 및 학습
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 로봇 테스트
print("학습 완료 후 테스트:")
obs = env.reset()
env.render()  # 초기 상태 렌더링
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break


# #model.save("ppo_mobile_robot")





import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MobileRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MobileRobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.grid_size = 7  # 그리드 크기 설정
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
                                                high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]), 
                                                dtype=np.float32)
        
        # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
        self.action_space = gym.spaces.Discrete(4)
        
        # 장애물 위치 설정 (가장자리 및 추가 장애물 배치)
        self.obstacles = [np.array([0, i]) for i in range(self.grid_size)] + \
                         [np.array([self.grid_size-1, i]) for i in range(self.grid_size)] + \
                         [np.array([i, 0]) for i in range(1, self.grid_size-1)] + \
                         [np.array([i, self.grid_size-1]) for i in range(1, self.grid_size-1)] + \
                         [np.array([1, 1]), np.array([1, 2]), np.array([2, 2]), np.array([1, 3]), 
                          np.array([1, 4]), np.array([4, 1]), np.array([4, 2])]

        # 로봇 시작 위치 및 목표 위치 랜덤 초기화
        self.robot_pos = self._generate_random_start()
        self.goal_pos = self._generate_random_goal()
    
    def _generate_random_goal(self):
        while True:
            goal_pos = np.random.randint(1, self.grid_size-1, size=2)
            if not any(np.array_equal(goal_pos, obs) for obs in self.obstacles) and not np.array_equal(goal_pos, self.robot_pos):
                return goal_pos
    
    def _generate_random_start(self):
        while True:
            start_pos = np.random.randint(1, self.grid_size-1, size=2)
            if not any(np.array_equal(start_pos, obs) for obs in self.obstacles):
                return start_pos

    def reset(self):
        # 로봇의 위치와 목표 위치를 랜덤하게 재설정
        self.robot_pos = self._generate_random_start()
        self.goal_pos = self._generate_random_goal()
        return self._get_obs()
    
    def step(self, action):
        # 로봇의 행동 적용
        if action == 0 and self.robot_pos[1] < self.grid_size-1:  # 위로 이동
            self.robot_pos[1] += 1
        elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
            self.robot_pos[1] -= 1
        elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
            self.robot_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.grid_size-1:  # 오른쪽으로 이동
            self.robot_pos[0] += 1
        
        # 보상 계산 및 완료 여부 확인
        reward = self._calculate_reward()
        done = np.array_equal(self.robot_pos, self.goal_pos)
        
        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        # 로봇의 위치와 목표물의 위치 반환
        return np.concatenate((self.robot_pos, self.goal_pos))
    
    def _calculate_reward(self):
        # 장애물에 충돌하면 큰 패널티
        for obstacle in self.obstacles:
            if np.array_equal(self.robot_pos, obstacle):
                return -10
        
        # 목표에 도달하면 큰 보상
        if np.array_equal(self.robot_pos, self.goal_pos):
            return 100
        
        # 목표로부터의 거리 감소에 따른 보상
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -distance_to_goal
        
        return reward
    
    def render(self, mode='human'):
        # 환경을 텍스트로 시각화
        env_map = np.zeros((self.grid_size, self.grid_size), dtype=str)
        env_map[:, :] = '.'
        env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
        env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
        for obstacle in self.obstacles:
            env_map[obstacle[1], obstacle[0]] = 'X'
        
        print("\n".join([" ".join(row) for row in env_map]))
        print()

# 환경 초기화 (render_mode 설정)
env = MobileRobotEnv(render_mode='human')
env = DummyVecEnv([lambda: env])

# PPO 모델 초기화 및 학습
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 로봇 테스트
print("학습 완료 후 테스트:")
obs = env.reset()
env.render()  # 초기 상태 렌더링
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # 각 단계마다 상태 렌더링
    if dones:
        print("Goal reached or obstacle hit, resetting environment.")
        obs = env.reset()


# #model.save("ppo_mobile_robot")




