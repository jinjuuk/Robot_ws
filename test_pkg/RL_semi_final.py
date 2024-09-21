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
                          np.array([1, 4]), np.array([4, 1]), np.array([4, 2]), np.array([2,1])]

        # 로봇, 목표 위치, 동적 장애물 초기화
        self.robot_pos = None
        self.goal_pos = None
        self.dynamic_obstacles = []

        # 초기 상태 설정
        self.reset()

    def _generate_random_position(self, exclude_positions):
        """ 주어진 위치들과 겹치지 않는 무작위 위치 생성 """
        while True:
            pos = np.random.randint(0, self.grid_size, size=2)
            if not any(np.array_equal(pos, ex_pos) for ex_pos in exclude_positions):
                return pos




    # def reset(self):
    #     # 목표 위치 설정
    #     self.goal_pos = self._generate_random_position(self.obstacles)
        
    #     # 동적 장애물 위치 설정 (목표 지점 및 정적 장애물과 겹치지 않음)
    #     self.dynamic_obstacles = [self._generate_random_position(self.obstacles + [self.goal_pos])]
        
    #     # 로봇 위치 설정 (목표 지점, 정적 장애물, 동적 장애물과 겹치지 않음)
    #     self.robot_pos = self._generate_random_position(self.obstacles + [self.goal_pos] + self.dynamic_obstacles)
        
    #     return self._get_obs()


    def reset(self):
        # 1. 모든 가능한 위치 집합 생성
        all_positions = {tuple(pos) for pos in np.ndindex(self.grid_size, self.grid_size)}
        
        # 2. 정적 장애물 위치 제외
        available_positions = all_positions - {tuple(obs) for obs in self.obstacles}
        
        # 3. 로봇 위치 설정
        self.robot_pos = np.array(random.sample(available_positions, 1)[0])
        
        # 4. 설정된 로봇 위치를 뺀 가능한 위치 집합 생성
        available_positions -= {tuple(self.robot_pos)}
        
        # 4. 목표 위치 설정
        self.goal_pos = np.array(random.sample(available_positions, 1)[0])
        
        # 5. 설정된 목표 위치를 뺀 가능한 위치 집합 생성
        available_positions -= {tuple(self.goal_pos)}
        
        # 6. 동적 장애물 위치 설정
        self.dynamic_obstacles = []
        dynamic_obstacle_pos = np.array(random.sample(available_positions, 1)[0])
        self.dynamic_obstacles.append(dynamic_obstacle_pos)
        
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




# 환경 초기화 (render_mode 설정)
env = MobileRobotEnv(render_mode='human')
env = DummyVecEnv([lambda: env])

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

#     for _ in range(100):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
        
#         # `info`는 리스트로 반환되므로 첫 번째 요소에 접근해야 함
#         info = info[0]

#         total_reward += rewards
#         total_distance_reward += info['distance_reward']
#         total_static_penalty += info['static_penalty']
#         total_dynamic_penalty += info['dynamic_penalty']
        
#         if info['static_penalty'] < 0:  # 정적 장애물에 부딪힘
#             collision_count_static += 1
#         if info['dynamic_penalty'] < 0:  # 동적 장애물에 부딪힘
#             collision_count_dynamic += 1

#         if dones:
#             break
    
#     # 학습마다 결과 저장
#     episode_rewards.append(total_reward)
#     distance_rewards.append(total_distance_reward)
#     static_penalties.append(total_static_penalty)
#     dynamic_penalties.append(total_dynamic_penalty)
#     collision_counts_static.append(collision_count_static)
#     collision_counts_dynamic.append(collision_count_dynamic)


# 학습
total_episodes = 10  # 예시로 10번의 에피소드 학습
for episode in range(total_episodes):
    print(f"에피소드 {episode + 1} 시작")
    
    # 환경 초기화 및 위치 확인
    obs = env.reset()
    
    # DummyVecEnv 내부의 실제 환경에 접근
    actual_env = env.envs[0]  # envs 리스트의 첫 번째 환경에 접근
    
    # 각 위치 출력 (확인용)
    print(f"  목표 위치: {actual_env.goal_pos}")
    print(f"  로봇 위치: {actual_env.robot_pos}")
    print(f"  동적 장애물 위치: {actual_env.dynamic_obstacles}")
    
    # 목표, 동적 장애물, 로봇 위치가 서로 겹치지 않도록 확인
    assert not np.array_equal(actual_env.goal_pos, actual_env.robot_pos), "로봇 위치가 목표 위치와 겹칩니다!"
    for dyn_obs in actual_env.dynamic_obstacles:
        assert not np.array_equal(actual_env.goal_pos, dyn_obs), "동적 장애물이 목표 위치와 겹칩니다!"
        assert not np.array_equal(actual_env.robot_pos, dyn_obs), "로봇 위치가 동적 장애물과 겹칩니다!"
    
    total_reward = 0
    total_distance_reward = 0
    total_static_penalty = 0
    total_dynamic_penalty = 0
    collision_count_static = 0
    collision_count_dynamic = 0

    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        
        # `info`는 리스트로 반환되므로 첫 번째 요소에 접근해야 함
        info = info[0]

        total_reward += rewards
        total_distance_reward += info['distance_reward']
        total_static_penalty += info['static_penalty']
        total_dynamic_penalty += info['dynamic_penalty']
        
        if info['static_penalty'] < 0:  # 정적 장애물에 부딪힘
            collision_count_static += 1
        if info['dynamic_penalty'] < 0:  # 동적 장애물에 부딪힘
            collision_count_dynamic += 1

        if dones:
            break

    # 에피소드 결과 출력 (확인용)
    print(f"  총 보상: {total_reward}")
    print(f"  거리 보상: {total_distance_reward}")
    print(f"  정적 장애물 충돌 패널티: {total_static_penalty}")
    print(f"  동적 장애물 충돌 패널티: {total_dynamic_penalty}")
    print(f"  정적 장애물 충돌 횟수: {collision_count_static}")
    print(f"  동적 장애물 충돌 횟수: {collision_count_dynamic}")
    print()
    print()









# 학습 종료 시간 기록
end_time = time.time()

# 학습 소요 시간 계산
elapsed_time = end_time - start_time


# model.save("ppo_RL_weights_100000_500")
