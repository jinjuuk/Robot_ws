# import gym
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.envs import DummyVecEnv
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv



# # 가상의 모바일 로봇 환경 (Gym에서 제공되거나 직접 구현)
# class MobileRobotEnv(gym.Env):

#     def __init__(self):
#         super(MobileRobotEnv, self).__init__()
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
#         self.action_space = gym.spaces.Discrete(4)  # 예: 4방향 이동

    
#     def reset(self):
#         # 초기 상태 설정
#         return self._get_obs()

    
#     def step(self, action):
#         # 로봇의 행동 적용
#         # 웨이포인트와 목표에 따른 보상 계산
#         done = False
#         reward = self._calculate_reward()
#         return self._get_obs(), reward, done, {}

    
#     def _get_obs(self):
#         # 상태 관측 반환
#         return np.array([0, 0, 0, 0])

    
#     def _calculate_reward(self):
#         # 보상 함수 정의
#         return 0


# # 환경 초기화
# env = MobileRobotEnv()
# env = DummyVecEnv([lambda: env])


# # PPO 모델 초기화 및 학습
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)


# # 학습된 모델로 로봇 테스트
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


# model.save("ppo_mobile_robot")




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

        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), 
                                                high=np.array([self.grid_size-1, self.grid_size-1, self.grid_size-1, self.grid_size-1]), 
                                                dtype=np.float32)
        
        self.action_space = gym.spaces.Discrete(4)
        
        # 장애물 위치 설정 (가장자리 및 추가 장애물 배치)
        self.obstacles = [np.array([0, i]) for i in range(self.grid_size)] + \
                         [np.array([self.grid_size-1, i]) for i in range(self.grid_size)] + \
                         [np.array([i, 0]) for i in range(1, self.grid_size-1)] + \
                         [np.array([i, self.grid_size-1]) for i in range(1, self.grid_size-1)] + \
                         [np.array([1, 1]), np.array([1, 2]), np.array([2, 2]), np.array([1, 3]), 
                          np.array([1, 4]), np.array([4, 1]), np.array([4, 2])]

        self.robot_pos = None
        self.goal_pos = None
        self.dynamic_obstacles = []

        # 모든 가능한 위치 집합 생성
        self.all_positions = self._generate_all_positions()

        self.reset()

    def _generate_all_positions(self):
        """ 모든 가능한 위치 집합 생성 """
        all_pos = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = np.array([x, y])
                if not any(np.array_equal(pos, obs) for obs in self.obstacles):
                    all_pos.append(pos)
        return all_pos

    def _select_random_position(self, available_positions):
        """ 주어진 위치 집합에서 무작위로 위치 선택 """
        return random.choice(available_positions).copy()  # NumPy 배열의 복사본을 반환

    def reset(self):
        # 1. 모든 가능한 위치 집합 생성 (이미 __init__에서 생성됨)
        available_positions = self.all_positions.copy()

        # 2. 목표 위치 설정
        self.goal_pos = self._select_random_position(available_positions)
        available_positions = [pos for pos in available_positions if not np.array_equal(pos, self.goal_pos)]

        # 3. 동적 장애물 위치 설정
        self.dynamic_obstacles = [self._select_random_position(available_positions)]
        available_positions = [pos for pos in available_positions if not np.array_equal(pos, self.dynamic_obstacles[0])]

        # 4. 로봇 위치 설정
        self.robot_pos = self._select_random_position(available_positions)
        
        return self._get_obs()

    def step(self, action):
        # 로봇의 행동 적용
        if action == 0 and self.robot_pos[1] < self.grid_size - 1:  # 위로 이동
            self.robot_pos[1] += 1
        elif action == 1 and self.robot_pos[1] > 0:  # 아래로 이동
            self.robot_pos[1] -= 1
        elif action == 2 and self.robot_pos[0] > 0:  # 왼쪽으로 이동
            self.robot_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.grid_size - 1:  # 오른쪽으로 이동
            self.robot_pos[0] += 1

        # 동적 장애물 이동
        self._move_dynamic_obstacles()
        
        # 보상 계산 및 완료 여부 확인
        reward, distance_reward, static_penalty, dynamic_penalty = self._calculate_reward()
        
        # 에피소드 종료 조건 확인
        done = False
        if static_penalty < 0:
            print("장애물에 충돌했습니다!")
            done = True
        elif dynamic_penalty < 0:
            print("동적 장애물에 충돌했습니다!")
            done = True
        elif np.array_equal(self.robot_pos, self.goal_pos):
            print("목표에 도달했습니다!")
            done = True

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, done, {
            "distance_reward": distance_reward,
            "static_penalty": static_penalty,
            "dynamic_penalty": dynamic_penalty,
        }
    
    def _move_dynamic_obstacles(self):
        """ 동적 장애물들을 무작위로 이동 """
        for dyn_obs in self.dynamic_obstacles:
            move_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # 상하좌우 무작위 이동
            new_pos = dyn_obs + move_direction
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:  # 이동 후 위치가 유효한 범위 내일 때만 이동
                if not any(np.array_equal(new_pos, obs) for obs in self.obstacles) and \
                   not np.array_equal(new_pos, self.goal_pos) and \
                   not np.array_equal(new_pos, self.robot_pos):  # 장애물, 목표, 로봇과 겹치지 않음
                    dyn_obs += move_direction

    def _get_obs(self):
        # 로봇의 위치와 목표물의 위치 반환
        return np.concatenate((self.robot_pos, self.goal_pos))
    
    def _calculate_reward(self):
        static_penalty = 0
        dynamic_penalty = 0

        # 장애물에 충돌하면 큰 패널티
        for obstacle in self.obstacles:
            if np.array_equal(self.robot_pos, obstacle):
                static_penalty = -10

        # 동적 장애물과 겹쳤을 때 패널티
        for dyn_obs in self.dynamic_obstacles:
            if np.array_equal(self.robot_pos, dyn_obs):  # 동적 장애물과 같은 위치에 있을 때
                dynamic_penalty = -20

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
        if np.array_equal(self.robot_pos, self.goal_pos) or \
           any(np.array_equal(self.robot_pos, dyn_obs) for dyn_obs in self.dynamic_obstacles) or \
           any(np.array_equal(self.robot_pos, obs) for obs in self.obstacles):
            env_map[self.robot_pos[1], self.robot_pos[0]] = 'C'
        else:
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

# 환경 초기화 함수
def make_env():
    return MobileRobotEnv(render_mode='human')

# 벡터화된 환경 생성
env = DummyVecEnv([make_env])

# PPO 모델 초기화 및 학습
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# 학습 시작 시간 기록
start_time = time.time()

# 학습마다 점수를 기록할 리스트
episode_rewards = []
distance_rewards = []
static_penalties = []
dynamic_penalties = []
collision_counts_static = []
collision_counts_dynamic = []

# 학습
total_episodes = 10  # 예시로 10번의 에피소드 학습
for episode in range(total_episodes):
    print(f"에피소드 {episode + 1} 시작")
    obs = env.reset()  # 환경 초기화
    total_reward = 0
    total_distance_reward = 0
    total_static_penalty = 0
    total_dynamic_penalty = 0
    collision_count_static = 0
    collision_count_dynamic = 0
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        
        info = info[0]
        done = dones[0]

        total_reward += rewards[0]
        total_distance_reward += info['distance_reward']
        total_static_penalty += info['static_penalty']
        total_dynamic_penalty += info['dynamic_penalty']
        
        if info['static_penalty'] < 0:
            collision_count_static += 1
        if info['dynamic_penalty'] < 0:
            collision_count_dynamic += 1

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
