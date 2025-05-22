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
from tqdm import trange
from drivers import DisparityExtender

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparameters
learning_rate = 0.0003
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 10000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'map_easy3'

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

class PrioritizedReplayBuffer():
    def __init__(self, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.maxlen = buffer_limit
        self.alpha = alpha
        self.pos = 0

    def put(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.maxlen:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.maxlen

    def sample(self, n, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), n, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # unpack
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in samples:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst, dtype=torch.long),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
            torch.tensor(weights, dtype=torch.float),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + 1e-6  # epsilon to avoid zero

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(56, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 6)
        else:
            out = self.forward(obs)
            coin = random.random()
            if coin < epsilon:
                return random.randint(0, 6)
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

def train_double_per(q, q_target, memory, optimizer, beta=0.4):
    for i in range(1):
        s, a, r, s_prime, done_mask, weights, indices = memory.sample(batch_size, beta=beta)

        # Double DQN
        q_out = q(s)
        q_a = q_out.gather(1, a)
        next_q_values = q(s_prime)
        next_actions = next_q_values.argmax(dim=1, keepdim=True)
        next_q_target = q_target(s_prime)
        max_q_prime = next_q_target.gather(1, next_actions)
        target = r + gamma * max_q_prime * done_mask

        td_errors = q_a - target.detach()
        loss = (F.smooth_l1_loss(q_a, target, reduction='none') * weights.unsqueeze(1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # priority update
        memory.update_priorities(indices, td_errors.squeeze().abs())

def preprocess_lidar(ranges):
    return np.array(ranges[::20])

STEER_LIMIT = 0.4189
N_STEER = 7
STEER_VALUES = np.linspace(-STEER_LIMIT, STEER_LIMIT, N_STEER)
SPEED_VALUES = [3.0]
def decode_action(action_idx):
    steer_idx = action_idx // len(SPEED_VALUES)
    speed_idx = action_idx % len(SPEED_VALUES)
    return STEER_VALUES[steer_idx], SPEED_VALUES[speed_idx]


def collect_expert_data(buffer, env, driver, n_episodes=10):
    poses = np.array([[0., 0., np.radians(270)]])
    for epi in trange(n_episodes, desc="Collecting expert data"):
        obs, r, done, info = env.reset(poses=poses)
        lidar = preprocess_lidar(obs['scans'][0])
        speed = np.array([obs['linear_vels_x'][0]])
        yaw = np.array([obs['poses_theta'][0]])
        s = np.concatenate([lidar, speed, yaw])
        done = False
        while not done:
            # Driver의 행동 결정
            ego_odom = {
                'pose_x': obs['poses_x'][0],
                'pose_y': obs['poses_y'][0],
                'pose_theta': obs['poses_theta'][0],
                'linear_vel_x': obs['linear_vels_x'][0],
                'linear_vel_y': obs['linear_vels_y'][0],
                'angular_vel_z': obs['ang_vels_z'][0],
            }
            scan = obs['scans'][0]
            speed_expert, steer_expert = driver.process_lidar(scan)
            
            # DQN action space와 매핑
            # steer_expert와 speed_expert를 가장 가까운 action index로 변환
            steer_idx = (np.abs(STEER_VALUES - steer_expert)).argmin()
            speed_idx = (np.abs(np.array(SPEED_VALUES) - speed_expert)).argmin()
            action_idx = steer_idx * len(SPEED_VALUES) + speed_idx

            actions = np.array([[steer_expert, speed_expert]])
            obs_prime, r, done, info = env.step(actions)
            lidar_prime = preprocess_lidar(obs_prime['scans'][0])
            speed_prime = np.array([obs_prime['linear_vels_x'][0]])
            yaw_prime = np.array([obs_prime['poses_theta'][0]])
            s_prime = np.concatenate([lidar_prime, speed_prime, yaw_prime])

            done_mask = 0.0 if done else 1.0
            buffer.put((s, action_idx, r, s_prime, done_mask))
            s = s_prime
            obs = obs_prime


def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK)

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = PrioritizedReplayBuffer()

    # === Expert Buffer Pre-fill ===
    expert_driver = DisparityExtender()
    collect_expert_data(memory, env, expert_driver, n_episodes=15)  # 원하는 만큼
    print(f"Expert prefill 완료: {memory.size()}개")

    poses = np.array([[0., 0., np.radians(270)]])
    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    speed = 3.0
    fastlap = 100000.0
    laptimes = []
    
    # === step 단위 학습을 위한 변수 추가 ===
    total_steps = 0
    target_update_steps = 5000  # target network를 5000 step마다 동기화

    for n_epi in range(10000):
        epsilon = max(0.01, 0.15 - 0.14 * (total_steps / 10000))  # 1만 step 동안 선형 감소
        obs, r, done, info = env.reset(poses=poses)
        lidar = preprocess_lidar(obs['scans'][0])
        speed = np.array([obs['linear_vels_x'][0]])
        yaw = np.array([obs['poses_theta'][0]])
        s = np.concatenate([lidar, speed, yaw])
        done = False

        laptime = 0.0

        while not done:
            actions = []
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer, speed = decode_action(a)
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            lidar_prime = preprocess_lidar(obs['scans'][0])
            speed_prime = np.array([obs['linear_vels_x'][0]])
            yaw_prime = np.array([obs['poses_theta'][0]])
            s_prime = np.concatenate([lidar_prime, speed_prime, yaw_prime])

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            # --- step 단위로 학습 진행 ---
            if memory.size() > train_start:
                # 1~4회 중 선택, 보통 1회면 충분
                train_double_per(q, q_target, memory, optimizer)

                # step 단위로 target network 업데이트
                if total_steps % target_update_steps == 0:
                    q_target.load_state_dict(q.state_dict())

            total_steps += 1

            if done:
                laptimes.append(laptime)
                lap = round(obs['lap_times'][0], 3)
                if int(obs['lap_counts'][0]) == 2:
                    if fastlap > lap:
                        torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
                            round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                        fastlap = lap
                    break

        if n_epi % print_interval == 0 and n_epi != 0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100))

    print('train finish')
    env.close()
    save_name = os.path.join(work_dir + '_' + RACETRACK, "laptimes_plot.png")
    plot_durations_save(laptimes, save_path=save_name)
    print(f"Laptime plot saved to: {save_name}")


# def main():
#     today = get_today()
#     work_dir = "./" + today
#     os.makedirs(work_dir + '_' + RACETRACK)

#     env = gym.make('f110_gym:f110-v0',
#                    map="{}/maps/{}".format(current_dir, RACETRACK),
#                    map_ext=".png", num_agents=1)
#     q = Qnet()
#     q_target = Qnet()
#     q_target.load_state_dict(q.state_dict())
#     memory = PrioritizedReplayBuffer()

#     poses = np.array([[0., 0., np.radians(270)]])
#     print_interval = 10
#     optimizer = optim.Adam(q.parameters(), lr=learning_rate)
#     speed = 3.0
#     fastlap = 10000.0
#     laptimes = []

#     for n_epi in range(10000):
#         epsilon = max(0.01, 0.11 - 0.1 * (n_epi / 10000))  # Linear annealing from 8% to 1%
#         obs, r, done, info = env.reset(poses=poses)
#         lidar = preprocess_lidar(obs['scans'][0])
#         speed = np.array([obs['linear_vels_x'][0]])
#         yaw = np.array([obs['poses_theta'][0]])
#         s = np.concatenate([lidar, speed, yaw])
#         done = False

#         laptime = 0.0

#         while not done:
#             actions = []
#             a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
#             steer, speed = decode_action(a)
#             actions.append([steer, speed])
#             actions = np.array(actions)
#             obs, r, done, info = env.step(actions)
#             lidar_prime = preprocess_lidar(obs['scans'][0])
#             speed_prime = np.array([obs['linear_vels_x'][0]])
#             yaw_prime = np.array([obs['poses_theta'][0]])
#             s_prime = np.concatenate([lidar_prime, speed_prime, yaw_prime])

#             done_mask = 0.0 if done else 1.0
#             memory.put((s, a, r, s_prime, done_mask))
#             s = s_prime

#             laptime += r
#             env.render(mode='human_fast')

#             if done:
#                 laptimes.append(laptime)
#                 # plot_durations(laptimes)
#                 lap = round(obs['lap_times'][0], 3)
#                 if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
#                     torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
#                         round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
#                     fastlap = lap
#                     break

#         if memory.size() > train_start:
#             train_double_per(q, q_target, memory, optimizer)

#         if n_epi % print_interval == 0 and n_epi != 0:
#             q_target.load_state_dict(q.state_dict())
#             print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
#                   .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100))

#     print('train finish')
#     env.close()
#     save_name = os.path.join(work_dir + '_' + RACETRACK, "laptimes_plot.png")
#     plot_durations_save(laptimes, save_path=save_name)
#     print(f"Laptime plot saved to: {save_name}")

def eval():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    poses = np.array([[0., 0., np.radians(90)]])
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        lidar = preprocess_lidar(obs['scans'][0])
        speed = np.array([obs['linear_vels_x'][0]])
        yaw = np.array([obs['poses_theta'][0]])
        s = np.concatenate([lidar, speed, yaw])

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []
            a = q.action(torch.from_numpy(s).float())
            steer, speed = decode_action(a)
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            lidar_prime = preprocess_lidar(obs['scans'][0])
            speed_prime = np.array([obs['linear_vels_x'][0]])
            yaw_prime = np.array([obs['poses_theta'][0]])
            s = np.concatenate([lidar_prime, speed_prime, yaw_prime])

            laptime += r
            env.render(mode='human_fast')

            if done:
                break
    env.close()
    
def plot_durations_save(laptimes, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path)
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == '__main__':
    main()
    # eval()
