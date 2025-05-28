import os
import sys
import collections
import random
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparameters
learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 7000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'map_easy3'


def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # self.fc1 = nn.Linear(405, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(270, 5)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 4)
        else:
            out = self.forward(obs)
            coin = random.random()
            if coin < epsilon:
                return random.randint(0, 4)
            else:
                return out.argmax().item()

    def action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()


def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)

    return np.array(ranges[eighth:-eighth: 3])

# disparity Extender 방식: difference를 구하는 함수
def get_differences(ranges):
    """ Gets the absolute difference between adjacent elements in
        in the LiDAR data and returns them in an array.
        Possible Improvements: replace for loop with numpy array arithmetic
    """
    differences = [0.]  # set first element to 0
    for i in range(1, len(ranges)):
        differences.append(abs(ranges[i] - ranges[i - 1]))
    return differences

# disparity Extender 방식: threshold를 넘는 disparity index를 구하는 함수
def get_disparities(differences, threshold):
    """ Gets the indexes of the LiDAR points that were greatly
        different to their adjacent point.
        Possible Improvements: replace for loop with numpy array arithmetic
    """
    disparities = []
    for index, difference in enumerate(differences):
        if difference > threshold:
            disparities.append(index)
    return disparities

def extend_disparities(disparities, ranges, car_width, extra_pct):
    """ For each pair of points we have decided have a large difference
        between them, we choose which side to cover (the opposite to
        the closer point), call the cover function, and return the
        resultant covered array.
        Possible Improvements: reduce to fewer lines
    """
    width_to_cover = (car_width / 2) * (1 + extra_pct / 100)
    for index in disparities:
        first_idx = index - 1
        points = ranges[first_idx:first_idx + 2]
        close_idx = first_idx + np.argmin(points)
        far_idx = first_idx + np.argmax(points)
        close_dist = ranges[close_idx]
        num_points_to_cover = get_num_points_to_cover(close_dist,
                                                            width_to_cover)
        cover_right = close_idx < far_idx
        ranges = cover_points(num_points_to_cover, close_idx,
                                    cover_right, ranges)
    return ranges

def get_num_points_to_cover(dist, width):
    radians_per_point = (2 * np.pi) / 1080 * 3 # 나는 3개씩 건너뛰니까 3을 곱해줌

    angle = 2 * np.arcsin(width / (2 * dist))
    num_points = int(np.ceil(angle / radians_per_point))
    return num_points

def cover_points(num_points, start_idx, cover_right, ranges):
    new_dist = ranges[start_idx]
    if cover_right:
        for i in range(num_points):
            next_idx = start_idx + 1 + i
            if next_idx >= len(ranges): break
            if ranges[next_idx] > new_dist:
                ranges[next_idx] = new_dist
    else:
        for i in range(num_points):
            next_idx = start_idx - 1 - i
            if next_idx < 0: break
            if ranges[next_idx] > new_dist:
                ranges[next_idx] = new_dist
    return ranges

def preprocess_lidar_all(ranges):
    DIFFERENCE_THRESHOLD = 2
    CAR_WIDTH = 0.31
    SAFETY_PERCENTAGE = 300
    proc_ranges = preprocess_lidar(ranges)
    differences = get_differences(proc_ranges)
    disparities = get_disparities(differences, DIFFERENCE_THRESHOLD)
    proc_ranges = extend_disparities(disparities, proc_ranges, CAR_WIDTH, SAFETY_PERCENTAGE)

    return proc_ranges


def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK)

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    poses = np.array([[0., 0., np.radians(270)]])
    # poses = np.array([[0.8007017, -0.2753365, 4.1421595]])

    # ---- Prefill ReplayBuffer with rule-based transitions ----
    poses = np.array([[0., 0., np.radians(270)]])
    for _ in range(2): # 에피소드 2번
        env_prefill = gym.make('f110_gym:f110-v0',
                            map="{}/maps/{}".format(current_dir, RACETRACK),
                            map_ext=".png", num_agents=1)
        obs, r, done, info = env_prefill.reset(poses=poses)
        s = preprocess_lidar_all(obs['scans'][0])
        done = False
        tmp_reward = 0.0
        while not done:
            # np.argmax로 가장 먼 곳 인덱스
            idx = np.argmax(s)
            rel = idx - 135   # 0~269에서 135 중심
            
            # 행동 결정: 대칭 (좌/우)
            if -4 <= rel <= 4:
                a = 2  # 직진
            elif 5 <= rel <= 9:
                a = 3  # 우회전
            elif rel >= 10:
                a = 4  # 강한 우회전
            elif -9 <= rel <= -5:
                a = 1  # 좌회전
            elif rel <= -10:
                a = 0  # 강한 좌회전
            else:
                a = 2  # 예외적으로 직진

            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 5.0
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4.0

            actions = np.array([[steer, speed]])
            obs2, r, done, info = env_prefill.step(actions)
            s_prime = preprocess_lidar_all(obs2['scans'][0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime
            # env_prefill.render(mode='human_fast')
            tmp_reward += r

        print('reward:', tmp_reward)
        env_prefill.close()
    print(f'Prefilled buffer: {memory.size()} transitions')
    # ---- Prefill 끝 ----

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    speed = 3.0
    fastlap = 10000.0
    laptimes = []
    total_rewards = []

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar_all(obs['scans'][0])
        done = False

        laptime = 0.0
        total_reward = 0.0

        while not done:
            # env.render(mode='human_fast')

            actions = []

            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 5.0
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4.0
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            # Disparity Extender 방식대로 보상 추가
            idx = np.argmax(s)
            rel = idx - 135   # 0~269에서 135 중심
            steer_real_angle  = steer * (180 / np.pi)  # 라디안 -> 도 변환
            if abs(rel - steer_real_angle) <= 3:
                r += 0.04
            elif abs(rel) < 15: # 차이 나는만큼 마이너스 보상
                r -= abs(rel - steer_real_angle) * 0.003
            if steer * rel < 0 and abs(rel) > 3:  # 돌려야 할 방향과 반대 방향으로 회전 돌리면
                r -= 0.05
            elif steer * rel > 0 and abs(rel) > 3:  # 돌려야 할 방향과 같은 방향으로 회전 돌리면
                r += 0.05
            if abs(rel) > 15 and abs(a) == 2 and rel * (a-2) * rel > 0: # 많이 꺾어야할때, 꺾어야할 방향으로 많이 꺾으면
                r += 0.05
            if abs(rel) < 3 and a != 2:  # 직진인데 회전하면
                r -= 0.02

            s_prime = preprocess_lidar_all(obs['scans'][0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            laptime += 0.01
            total_reward += r
            env.render(mode='human_fast')

            if done:
                total_rewards.append(total_reward)
                # plot_durations(total_rewards)
                lap = round(obs['lap_times'][0], 3)
                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                    torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, laptime : {:.1f}, reward : {:.1f}, lap_counts: {}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime, total_reward, obs['lap_counts'][0], memory.size(), epsilon * 100))

    print('train finish')
    env.close()


def eval():
    RACETRACK = 'map_easy3'
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("{}/2025-05-28_18-41-01_map_easy3_rulebase/fast-model41.2_6429.pt".format(current_dir)))
    poses = np.array([[0., 0., np.radians(270)]])
    # speed = 3.0
    for t in range(1):
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar_all(obs['scans'][0])

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 5.0
            elif a == 1 or a == 3:
                speed = 4.5
            else:
                speed = 4.0
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar_all(obs['scans'][0])

            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                break
    env.close()


if __name__ == '__main__':
    # main()
    eval()
