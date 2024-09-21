# import gym
# import numpy as np
# import time
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# import random

# class MobileRobotEnv(gym.Env):
#     def __init__(self, render_mode=None):
#         super(MobileRobotEnv, self).__init__()
        
#         self.render_mode = render_mode
#         self.grid_size = 7  # 7x7 그리드

#         # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
#         self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
#                                                 high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]), 
#                                                 dtype=np.float32)
        
#         # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
#         self.action_space = gym.spaces.Discrete(4)
        
#         # 장애물 위치 설정 (가장자리 및 추가 장애물 배치)
#         self.obstacles = [np.array([0, i]) for i in range(self.grid_size)] + \
#                          [np.array([self.grid_size-1, i]) for i in range(self.grid_size)] + \
#                          [np.array([i, 0]) for i in range(1, self.grid_size-1)] + \
#                          [np.array([i, self.grid_size-1]) for i in range(1, self.grid_size-1)] + \
#                          [np.array([1, 1]), np.array([1, 2]), np.array([2, 2]), np.array([1, 3]), 
#                           np.array([1, 4]), np.array([4, 1]), np.array([4, 2]), np.array([2,1])]

#         # 로봇, 목표 위치, 동적 장애물 초기화
#         self.robot_pos = None
#         self.goal_pos = None
#         self.dynamic_obstacles = []

#         # 초기 상태 설정
#         self.reset()

#     def _generate_random_position(self, exclude_positions):
#         """ 주어진 위치들과 겹치지 않는 무작위 위치 생성 """
#         while True:
#             pos = np.random.randint(0, self.grid_size, size=2)
#             if not any(np.array_equal(pos, ex_pos) for ex_pos in exclude_positions):
#                 return pos


#     def reset(self):
#         # 목표 위치 설정
#         self.goal_pos = self._generate_random_position(self.obstacles)
        
#         # 동적 장애물 위치 설정 (목표 지점 및 정적 장애물과 겹치지 않음)
#         self.dynamic_obstacles = [self._generate_random_position(self.obstacles + [self.goal_pos])]
        
#         # 로봇 위치 설정 (목표 지점, 정적 장애물, 동적 장애물과 겹치지 않음)
#         self.robot_pos = self._generate_random_position(self.obstacles + [self.goal_pos] + self.dynamic_obstacles)
        
#         # 웨이포인트 초기화
#         self.waypoints = [tuple(self.robot_pos)]
        
#         return self._get_obs()
 

#     def step(self, action):
#         prev_pos = self.robot_pos.copy()
#         # 로봇의 행동 적용
#         if action == 0 and self.robot_pos[1] < self.grid_size - 1:  # 위로 이동
#             self.robot_pos[1] += 1
#         elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
#             self.robot_pos[1] -= 1
#         elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
#             self.robot_pos[0] -= 1
#         elif action == 3 and self.robot_pos[0] < self.grid_size - 1:  # 오른쪽으로 이동
#             self.robot_pos[0] += 1

#         # 로봇이 이동한 경우에만 웨이포인트를 추가
#         if not np.array_equal(prev_pos, self.robot_pos):
#             self.waypoints.append(self.robot_pos.copy())

#         # 동적 장애물 이동
#         self._move_dynamic_obstacles()
        
#         # 보상 계산 및 완료 여부 확인
#         reward, distance_reward, static_penalty, dynamic_penalty = self._calculate_reward()
        
#         # 에피소드 종료 조건 확인 -> collision 조건 추가
#         done = False
#         success = False
#         collision = False
#         if static_penalty < 0:
#             print("정적 장애물에 충돌했습니다!")
#             done = True
#             collision = True
#         elif dynamic_penalty < 0:
#             print("동적 장애물에 충돌했습니다!")
#             done = True
#             collision = True
#         elif np.array_equal(self.robot_pos, self.goal_pos):
#             print("목표에 도달했습니다!")
#             done = True
#             success = True
            

#         if self.render_mode == 'human':
#             self.render()

#         return self._get_obs(), reward, done, {
#             "distance_reward": distance_reward,
#             "static_penalty": static_penalty,
#             "dynamic_penalty": dynamic_penalty,
#             "success": success,
#             "collision": collision
#         }
    
#     def _move_dynamic_obstacles(self):
#         """ 동적 장애물들을 무작위로 이동 """
#         for dyn_obs in self.dynamic_obstacles:
#             move_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # 상하좌우 무작위 이동
#             new_pos = dyn_obs + move_direction
#             if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:  # 이동 후 위치가 유효한 범위 내일 때만 이동
#                 if not any(np.array_equal(new_pos, obs) for obs in self.obstacles) and \
#                    not np.array_equal(new_pos, self.goal_pos) and \
#                    not np.array_equal(new_pos, self.robot_pos):  # 장애물, 목표, 로봇과 겹치지 않음
#                     dyn_obs += move_direction

#     def _get_obs(self):
#         # 로봇의 위치와 목표물의 위치 반환
#         return np.concatenate((self.robot_pos, self.goal_pos))
    
#     def _calculate_reward(self):
#         static_penalty = 0
#         dynamic_penalty = 0

#         # 장애물에 충돌하면 큰 패널티
#         for obstacle in self.obstacles:
#             if np.array_equal(self.robot_pos, obstacle):
#                 static_penalty = -10

#         # 동적 장애물과 겹쳤을 때 패널티
#         for dyn_obs in self.dynamic_obstacles:
#             if np.array_equal(self.robot_pos, dyn_obs):  # 동적 장애물과 같은 위치에 있을 때
#                 dynamic_penalty = -20

#         # 목표에 도달하면 큰 보상
#         if np.array_equal(self.robot_pos, self.goal_pos):
#             return 100, 0, static_penalty, dynamic_penalty

#         # 목표로부터의 거리 감소에 따른 보상
#         distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
#         distance_reward = -distance_to_goal
        
#         total_reward = distance_reward + static_penalty + dynamic_penalty
#         return total_reward, distance_reward, static_penalty, dynamic_penalty
    
#     def render(self, mode='human'):
#         # 환경을 텍스트로 시각화
#         env_map = np.zeros((self.grid_size, self.grid_size), dtype=str)
#         env_map[:, :] = '.'
        
#         # 중첩된 위치 처리
#         # if np.array_equal(self.robot_pos, self.goal_pos) or \
#         #    any(np.array_equal(self.robot_pos, dyn_obs) for dyn_obs in self.dynamic_obstacles) or \
#         #    any(np.array_equal(self.robot_pos, obs) for obs in self.obstacles):
#         #     env_map[self.robot_pos[1], self.robot_pos[0]] = 'C'
#         # else:
#         env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
        
#         if not np.array_equal(self.goal_pos, self.robot_pos):
#             env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
        
#         for obstacle in self.obstacles:
#             if not np.array_equal(obstacle, self.robot_pos):
#                 env_map[obstacle[1], obstacle[0]] = 'X'
        
#         for dyn_obs in self.dynamic_obstacles:
#             if not np.array_equal(dyn_obs, self.robot_pos):
#                 env_map[dyn_obs[1], dyn_obs[0]] = 'D'
        
#         print("\n".join([" ".join(row) for row in env_map]))
#         print()


# # 환경 초기화 (render_mode 설정)
# env = MobileRobotEnv(render_mode='human')
# env = DummyVecEnv([lambda: env])

# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# # 학습 시작 시간 기록
# start_time = time.time()

# # 학습마다 점수를 기록할 리스트
# episode_rewards = []
# distance_rewards = []
# static_penalties = []
# dynamic_penalties = []
# collision_counts_static = []
# collision_counts_dynamic = []
# waypoints_list = []  # 각 에피소드의 웨이포인트를 저장할 리스트
# episode_results = []

# # 학습
# total_episodes = 10  # 예시로 10번의 에피소드 학습
# for episode in range(total_episodes):
#     print(f"에피소드 {episode + 1} 시작")
#     obs = env.reset()
#     total_reward = 0
#     total_distance_reward = 0
#     total_static_penalty = 0
#     total_dynamic_penalty = 0
#     collision_count_static = 0
#     collision_count_dynamic = 0
#     done = False
#     episode_waypoints = []  # 현재 에피소드의 웨이포인트

#     while not done:
#         action, states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
        
#         info = info[0]  # DummyVecEnv에서는 첫 번째 환경의 정보만 사용
#         done = dones[0]

#         # 현재 위치를 웨이포인트로 추가
#         current_pos = env.envs[0].robot_pos
#         episode_waypoints.append(np.array(current_pos))

#         total_reward += rewards[0]
#         total_distance_reward += info['distance_reward']
#         total_static_penalty += info['static_penalty']
#         total_dynamic_penalty += info['dynamic_penalty']
        
#         if info['static_penalty'] < 0:
#             collision_count_static += 1
#         if info['dynamic_penalty'] < 0:
#             collision_count_dynamic += 1

#         if done:
#             if info["success"]:
#                 episode_results.append("성공")
#                 waypoints_list.append(episode_waypoints)
#             elif collision_count_static > 0 or collision_count_dynamic > 0:
#                 episode_results.append("충돌함!")
#                 waypoints_list.append([])  # 충돌 시 웨이포인트는 저장하지 않음
#             else:
#                 episode_results.append("실패")
#                 waypoints_list.append(episode_waypoints)
#             break
    
#     # 학습마다 결과 저장
#     episode_rewards.append(total_reward)
#     distance_rewards.append(total_distance_reward)
#     static_penalties.append(total_static_penalty)
#     dynamic_penalties.append(total_dynamic_penalty)
#     collision_counts_static.append(collision_count_static)
#     collision_counts_dynamic.append(collision_count_dynamic)

# # 학습 종료 시간 기록
# end_time = time.time()

# # 학습 소요 시간 계산
# elapsed_time = end_time - start_time

# # 결과 출력
# print("학습 결과 요약")
# print(f"총 걸린 시간: {elapsed_time:.2f}초")
# print(f"학습마다의 최종 점수: {episode_rewards}")
# print(f"시간에 따른 점수: {episode_rewards}")
# print(f"거리 보상 점수: {distance_rewards}")
# print(f"정적 장애물에 대한 패널티 점수: {static_penalties}")
# print(f"동적 장애물에 대한 패널티 점수: {dynamic_penalties}")
# print(f"정적 장애물에 부딪힌 횟수: {collision_counts_static}")
# print(f"동적 장애물에 부딪힌 횟수: {collision_counts_dynamic}")

# # 각 에피소드의 웨이포인트 출력
# for i, (waypoints, result) in enumerate(zip(waypoints_list, episode_results)):
#     print(f"에피소드 {i+1}의 결과: {result}")
#     if waypoints:
#         for wp in waypoints:
#             print(f"np.array([{wp[0]}, {wp[1]}])", end=", ")
#         print()

# # model.save("new_jeans")









import gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random

class MobileRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(MobileRobotEnv, self).__init__()
        self.render_mode = render_mode
        self.grid_size = 7  # 7x7 그리드

        # 상태 공간: 로봇의 위치(x, y), 목표물의 위치(x, y)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]),
            dtype=np.float32
        )

        # 행동 공간: 상하좌우 이동 (0: 위, 1: 아래, 2: 왼쪽, 3: 오른쪽)
        self.action_space = gym.spaces.Discrete(4)

        # 장애물 위치 설정
        self.obstacles = self._set_obstacles()

        # 로봇, 목표 위치, 동적 장애물 초기화
        self.robot_pos = None
        self.goal_pos = None
        self.dynamic_obstacles = []
        

        # 초기 상태 설정
        self.reset()

    def _set_obstacles(self):
        # 가장자리 장애물 설정
        obstacles = ([np.array([0, i]) for i in range(self.grid_size)] +
                     [np.array([self.grid_size-1, i]) for i in range(self.grid_size)] +
                     [np.array([i, 0]) for i in range(1, self.grid_size-1)] +
                     [np.array([i, self.grid_size-1]) for i in range(1, self.grid_size-1)])
        
        # 추가 장애물 설정
        additional_obstacles = [
            np.array([1, 1]), np.array([1, 2]), np.array([2, 2]), np.array([1, 3]),
            np.array([1, 4]), np.array([4, 1]), np.array([4, 2]), np.array([2, 1])
        ]
        
        return obstacles + additional_obstacles

    def _generate_random_position(self, exclude_positions):
        """주어진 위치들과 겹치지 않는 무작위 위치 생성"""
        available_positions = []
        for x in range(1, self.grid_size-1):
            for y in range(1, self.grid_size-1):
                pos = np.array([x, y])
                if not any(np.array_equal(pos, ex_pos) for ex_pos in exclude_positions):
                    available_positions.append(pos)
        
        if not available_positions:
            raise ValueError("No available positions left")
        
        return available_positions[np.random.choice(len(available_positions))]

    def reset(self):
        # 모든 정적 장애물 위치
        all_obstacles = [tuple(obs) for obs in self.obstacles]

        # 목표 위치 설정
        self.goal_pos = self._generate_random_position(all_obstacles)
        
        # 로봇 위치 설정 (목표 지점, 정적 장애물과 겹치지 않음)
        while True:
            self.robot_pos = self._generate_random_position(all_obstacles + [tuple(self.goal_pos)])
            # 로봇과 목표 사이의 거리가 최소 2칸 이상이 되도록 함
            if np.linalg.norm(np.array(self.robot_pos) - np.array(self.goal_pos)) >= 2:
                break
        
        # 동적 장애물 위치 설정 (목표 지점, 정적 장애물, 로봇과 겹치지 않음)
        while True:
            dynamic_obs_pos = self._generate_random_position(all_obstacles + [tuple(self.goal_pos), tuple(self.robot_pos)])
            # 동적 장애물과 로봇 사이의 거리가 최소 2칸 이상이 되도록 함
            if np.linalg.norm(np.array(dynamic_obs_pos) - np.array(self.robot_pos)) >= 2:
                self.dynamic_obstacles = [dynamic_obs_pos]
                break
        
        # 웨이포인트 초기화
        self.waypoints = [tuple(self.robot_pos)]
        
        return self._get_obs()
 

    def step(self, action):
        prev_pos = self.robot_pos.copy()
        # 로봇의 행동 적용
        if action == 0 and self.robot_pos[1] < self.grid_size - 1:  # 위로 이동
            self.robot_pos[1] += 1
        elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
            self.robot_pos[1] -= 1
        elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
            self.robot_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.grid_size - 1:  # 오른쪽으로 이동
            self.robot_pos[0] += 1

        # 로봇이 이동한 경우에만 웨이포인트를 추가
        if not np.array_equal(prev_pos, self.robot_pos):
            self.waypoints.append(self.robot_pos.copy())

        # 동적 장애물 이동
        self._move_dynamic_obstacles()
        
        # 보상 계산 및 완료 여부 확인
        reward, distance_reward, static_penalty, dynamic_penalty = self._calculate_reward()
        
        # 에피소드 종료 조건 확인 -> collision 조건 추가
        done = False
        success = False
        collision = False
        if static_penalty < 0:
            print("정적 장애물에 충돌했습니다!")
            done = True
            collision = True
        elif dynamic_penalty < 0:
            print("동적 장애물에 충돌했습니다!")
            done = True
            collision = True
        elif np.array_equal(self.robot_pos, self.goal_pos):
            print("목표에 도달했습니다!")
            done = True
            success = True
            

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, done, {
            "distance_reward": distance_reward,
            "static_penalty": static_penalty,
            "dynamic_penalty": dynamic_penalty,
            "success": success,
            "collision": collision
        }
    
    # def _move_dynamic_obstacles(self):
    #     """ 동적 장애물들을 무작위로 이동 """
    #     for dyn_obs in self.dynamic_obstacles:
    #         move_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # 상하좌우 무작위 이동
    #         new_pos = dyn_obs + move_direction
    #         if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:  # 이동 후 위치가 유효한 범위 내일 때만 이동
    #             if not any(np.array_equal(new_pos, obs) for obs in self.obstacles) and \
    #                not np.array_equal(new_pos, self.goal_pos) and \
    #                not np.array_equal(new_pos, self.robot_pos):  # 장애물, 목표, 로봇과 겹치지 않음
    #                 dyn_obs += move_direction

    def _move_dynamic_obstacles(self):
        #동적 장애물들을 무작위로 이동#
        new_positions = []
        for dyn_obs in self.dynamic_obstacles:
            move_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # 상하좌우 무작위 이동
            new_pos = dyn_obs + np.array(move_direction)
            
            # 이동 후 위치가 유효한 범위 내이고, 장애물, 목표, 로봇, 다른 동적 장애물과 겹치지 않을 때만 이동
            if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size and
                not any(np.array_equal(new_pos, obs) for obs in self.obstacles) and
                not np.array_equal(new_pos, self.goal_pos) and
                not np.array_equal(new_pos, self.robot_pos) and
                not any(np.array_equal(new_pos, other_new_pos) for other_new_pos in new_positions)):
                new_positions.append(new_pos)
            else:
                new_positions.append(dyn_obs)  # 이동할 수 없으면 현재 위치 유지
        
        self.dynamic_obstacles = new_positions

    def _get_obs(self):
        # 로봇의 위치와 목표물의 위치 반환
        return np.concatenate((self.robot_pos, self.goal_pos))
    
    def _calculate_reward(self):
        static_penalty = 0
        dynamic_penalty = 0

        # 장애물에 충돌하면 큰 패널티
        for obstacle in self.obstacles:
            if np.array_equal(self.robot_pos, obstacle):
                static_penalty = -20

        # 동적 장애물과 겹쳤을 때 패널티
        for dyn_obs in self.dynamic_obstacles:
            if np.array_equal(self.robot_pos, dyn_obs):  # 동적 장애물과 같은 위치에 있을 때
                dynamic_penalty = -40

        # 목표에 도달하면 큰 보상
        if np.array_equal(self.robot_pos, self.goal_pos):
            return 100, 0, static_penalty, dynamic_penalty

        # 목표로부터의 거리 감소에 따른 보상
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        distance_reward = -distance_to_goal
        
        total_reward = distance_reward + static_penalty + dynamic_penalty
        return total_reward, distance_reward, static_penalty, dynamic_penalty
    
    def render(self, mode='human'):
        # 환경을 텍스트로 시각화
        env_map = np.zeros((self.grid_size, self.grid_size), dtype=str)
        env_map[:, :] = '.'
        
        # 중첩된 위치 처리
        # if np.array_equal(self.robot_pos, self.goal_pos) or \
        #    any(np.array_equal(self.robot_pos, dyn_obs) for dyn_obs in self.dynamic_obstacles) or \
        #    any(np.array_equal(self.robot_pos, obs) for obs in self.obstacles):
        #     env_map[self.robot_pos[1], self.robot_pos[0]] = 'C'
        # else:
        env_map[self.robot_pos[1], self.robot_pos[0]] = 'R'
        
        if not np.array_equal(self.goal_pos, self.robot_pos):
            env_map[self.goal_pos[1], self.goal_pos[0]] = 'G'
        
        for obstacle in self.obstacles:
            if not np.array_equal(obstacle, self.robot_pos):
                env_map[obstacle[1], obstacle[0]] = 'X'
        
        for dyn_obs in self.dynamic_obstacles:
            if not np.array_equal(dyn_obs, self.robot_pos):
                env_map[dyn_obs[1], dyn_obs[0]] = 'D'
        
        print("\n".join([" ".join(row) for row in env_map]))
        print()


# 환경 초기화 (render_mode 설정)
env = MobileRobotEnv(render_mode='human')
env = DummyVecEnv([lambda: env])

# PPO 모델 초기화 및 학습
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# 학습 시작 시간 기록
start_time = time.time()

# 학습마다 점수를 기록할 리스트
episode_rewards = []
distance_rewards = []
static_penalties = []
dynamic_penalties = []
collision_counts_static = []
collision_counts_dynamic = []
waypoints_list = []  # 각 에피소드의 웨이포인트를 저장할 리스트
episode_results = []

# 학습
total_episodes = 10  # 예시로 10번의 에피소드 학습
for episode in range(total_episodes):
    print(f"에피소드 {episode + 1} 시작")
    obs = env.reset()
    total_reward = 0
    total_distance_reward = 0
    total_static_penalty = 0
    total_dynamic_penalty = 0
    collision_count_static = 0
    collision_count_dynamic = 0
    done = False
    episode_waypoints = []  # 현재 에피소드의 웨이포인트

    while not done:
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        
        info = info[0]  # DummyVecEnv에서는 첫 번째 환경의 정보만 사용
        done = dones[0]

        # 현재 위치를 웨이포인트로 추가
        current_pos = env.envs[0].robot_pos
        episode_waypoints.append(np.array(current_pos))

        total_reward += rewards[0]
        total_distance_reward += info['distance_reward']
        total_static_penalty += info['static_penalty']
        total_dynamic_penalty += info['dynamic_penalty']
        
        if info['static_penalty'] < 0:
            collision_count_static += 1
        if info['dynamic_penalty'] < 0:
            collision_count_dynamic += 1

        if done:
            if info["success"]:
                episode_results.append("성공")
                waypoints_list.append(episode_waypoints)
            elif collision_count_static > 0 or collision_count_dynamic > 0:
                episode_results.append("충돌함!")
                waypoints_list.append([])  # 충돌 시 웨이포인트는 저장하지 않음
            else:
                episode_results.append("실패")
                waypoints_list.append(episode_waypoints)
            break
    
    # 학습마다 결과 저장
    episode_rewards.append(total_reward)
    distance_rewards.append(total_distance_reward)
    static_penalties.append(total_static_penalty)
    dynamic_penalties.append(total_dynamic_penalty)
    collision_counts_static.append(collision_count_static)
    collision_counts_dynamic.append(collision_count_dynamic)

# 학습 종료 시간 기록
end_time = time.time()

# 학습 소요 시간 계산
elapsed_time = end_time - start_time

# 결과 출력
print("학습 결과 요약")
print(f"총 걸린 시간: {elapsed_time:.2f}초")
print(f"학습마다의 최종 점수: {episode_rewards}")
print(f"시간에 따른 점수: {episode_rewards}")
print(f"거리 보상 점수: {distance_rewards}")
print(f"정적 장애물에 대한 패널티 점수: {static_penalties}")
print(f"동적 장애물에 대한 패널티 점수: {dynamic_penalties}")
print(f"정적 장애물에 부딪힌 횟수: {collision_counts_static}")
print(f"동적 장애물에 부딪힌 횟수: {collision_counts_dynamic}")

# 각 에피소드의 웨이포인트 출력
for i, (waypoints, result) in enumerate(zip(waypoints_list, episode_results)):
    print(f"에피소드 {i+1}의 결과: {result}")
    if waypoints:
        for wp in waypoints:
            print(f"np.array([{wp[0]}, {wp[1]}])", end=", ")
        print()

# model.save("new_jeans")








