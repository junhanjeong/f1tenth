import os
import sys
import random
import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from tqdm import trange

# Hyperparameters
LR = 0.00042
GAMMA = 0.98
BUFFER_SIZE = 50000
BATCH_SIZE = 32
TRAIN_START = 10000
TARGET_UPDATE_FREQ = 500

# LIDAR & Action parameters
STEER_LIMIT = 0.4189
N_STEER = 3
STEER_VALUES = np.linspace(-STEER_LIMIT, STEER_LIMIT, N_STEER)
SPEED = 3.0
MAP_NAME = 'map_easy3'


def get_timestamp():
    now = time.localtime()
    return time.strftime("%Y%m%d_%H%M%S", now)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(np.array(weights), dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td.item()) + 1e-6


# --- Preprocessing ---
def preprocess_lidar(ranges):
    """
    Convert raw 1080-point LiDAR over 270deg -> 180deg -> 20-group mean -> min-max scale to [0,1]
    """
    n = len(ranges)
    start = n // 6
    end = n - start
    sliced = np.array(ranges[start:end])
    group_size = len(sliced) // 20
    grouped = []
    max_dist = 10.0
    for i in range(20):
        seg = sliced[i*group_size:(i+1)*group_size]
        seg = seg[seg < max_dist]
        avg = seg.mean() if len(seg) > 0 else max_dist
        grouped.append(avg)
    arr = np.array(grouped)
    return np.clip(arr, 0, max_dist) / max_dist


def make_state(prev, curr):
    """
    Stack two consecutive 20-d LiDAR arrays -> shape (2,20)
    """
    return np.stack([prev, curr], axis=0)


# --- Reward shaping ---
def dqn_reward(action_idx, collision):
    if collision:
        return -1.5
    elif action_idx == 1:
        return 0.15
    else:
        return 0.05


def clip_reward(r):
    return float(np.clip(r, -1.0, 0.3))


# --- Q-Network ---
class QNet1D(nn.Module):
    def __init__(self, input_channels=2, input_length=20, num_actions=3):
        super().__init__()
        # conv1d expects (batch, channels, length)
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        conv_out_len = input_length - 4 + 1
        conv_out_len = conv_out_len - 2 + 1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * conv_out_len, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(N_STEER)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.forward(s)
            return q.argmax(1).item()


# --- Training Loop ---
def main():
    timestamp = get_timestamp()
    work_dir = f"./{timestamp}_{MAP_NAME}"
    os.makedirs(work_dir, exist_ok=True)

    env = gym.make('f110_gym:f110-v0',
                   map=f"{os.getcwd()}/maps/{MAP_NAME}",
                   map_ext=".png", num_agents=1)

    q_net = QNet1D()
    q_target = QNet1D()
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    memory = PrioritizedReplayBuffer(BUFFER_SIZE)

    total_steps = 0
    beta_start = 0.4
    epsilon = 1.0

    for episode in range(10000):
        epsilon = max(0.1, epsilon * 0.998)
        obs, _, _, _ = env.reset(poses=np.array([[0., 0., np.radians(270)]]))
        lidar0 = preprocess_lidar(obs['scans'][0])
        prev = lidar0.copy()
        curr = lidar0.copy()
        state = make_state(prev, curr)
        done = False
        laptime = 0.0

        while not done:
            action = q_net.act(state, epsilon)
            steer = STEER_VALUES[action]
            action_np = np.array([[steer, SPEED]])
            obs2, r, done, info = env.step(action_np)
            lidar1 = preprocess_lidar(obs2['scans'][0])

            # 왼쪽 트랙에 붙으면 보상
            OPTIMAL_LEFT_DISTANCE_MAX = 0.16
            left_lidar_avg = np.mean(lidar1[0:5])
            if left_lidar_avg < OPTIMAL_LEFT_DISTANCE_MAX:
                r += 0.05

            next_state = make_state(curr, lidar1) if not done else make_state(curr, curr)
            collision = bool(obs2['collisions'][0])
            r += clip_reward(dqn_reward(action, collision))
            memory.add((state, action, r, next_state, 0.0 if done else 1.0))
            state = next_state
            prev, curr = curr, lidar1
            total_steps += 1
            laptime += r

            if len(memory) > TRAIN_START:
                beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / 10000)
                s_b, a_b, r_b, s2_b, d_b, w_b, idxs = memory.sample(BATCH_SIZE, beta)
                q_out = q_net(s_b)
                q_val = q_out.gather(1, a_b)
                next_actions = q_net(s2_b).argmax(1, keepdim=True)
                q_target_val = q_target(s2_b).gather(1, next_actions)
                target = r_b + GAMMA * q_target_val * d_b
                td_errors = q_val - target.detach()
                loss = (w_b * F.smooth_l1_loss(q_val, target, reduction='none')).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                memory.update_priorities(idxs, td_errors.abs().squeeze())
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    q_target.load_state_dict(q_net.state_dict())

            env.render(mode='human_fast')

        if episode % 10 == 0:
            print(f"Episode {episode} | LapReward: {laptime:.2f} | Buffer: {len(memory)} | Eps: {epsilon:.3f}")

    torch.save(q_net.state_dict(), os.path.join(work_dir, "qnet_final.pt"))
    env.close()


if __name__ == '__main__':
    main()
