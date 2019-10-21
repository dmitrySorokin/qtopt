from torch import optim
from torch import nn
from cem import CEM
from qnet import QNetwork

import torch
import numpy as np


class QTOpt(object):
    def __init__(
            self, replay_buffer, obs_shape, action_space,
            q_lr=3e-4, cem_update_itr=2,
            select_num=6, num_samples=64):
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.device = torch.device('cpu')

        num_inputs = obs_shape[0]
        num_outputs = action_space.shape[0]

        self.qnet = QNetwork(num_inputs, num_outputs)
        self.target_qnet1 = QNetwork(num_inputs, num_outputs)
        self.target_qnet2 = QNetwork(num_inputs, num_outputs)

        self.cem = CEM(theta_dim=num_outputs)  # cross-entropy method for updating

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0

    def to(self, device):
        self.device = device
        self.qnet = self.qnet.to(device)
        self.target_qnet1 = self.target_qnet1.to(device)
        self.target_qnet2 = self.target_qnet2.to(device)
        return self

    def update(self, batch_size, gamma=0.9, soft_tau=1e-4, update_delay=10000):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        self.step_cnt += 1

        state = torch.from_numpy(state).to(self.device)
        next_state_ = torch.from_numpy(next_state).to(self.device)
        action = torch.from_numpy(action).to(self.device)

        # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)

        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predict_q = self.qnet(state, action)  # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a')

        new_next_action = []
        for i in range(batch_size):  # batch of states, use them one by one, to prevent the lack of memory
            new_next_action.append(self.cem_optimal_action(
                np.expand_dims(next_state[i], axis=0)
            ))
        new_next_action = torch.FloatTensor(new_next_action).to(self.device)

        target_q_min = torch.min(
            self.target_qnet1(next_state_, new_next_action),
            self.target_qnet2(next_state_, new_next_action)
        )

        target_q = reward + (1 - done) * gamma * target_q_min

        # MSE loss, note that original paper uses cross-entropy loss
        q_loss = ((predict_q - target_q.detach()) ** 2).mean()

        ######################
        # value_target = torch.from_numpy(value_target).float().to(device)
        # my_loss = ((predict_q - value_target)**2).mean()
        # q_loss += my_loss
        #####################

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 0.5)
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1 = self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2 = self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)

        return q_loss.detach().cpu().numpy()

    def cem_optimal_action(self, state):
        """ evaluate action wrt Q(s,a) to select the optimal using CEM """
        cuda_states = torch.from_numpy(np.vstack([state] * self.num_samples)).to(self.device)
        # every time use a new cem, cem is only for deriving the argmax_a'
        self.cem.initialize()
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            q_values = self.target_qnet1(
                cuda_states,
                torch.from_numpy(actions).float().to(self.device)
            ).detach().cpu().numpy().reshape(-1)  # 2 dim to 1 dim
            max_idx = q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_actions = actions[idx]
            _, _ = self.cem.update(selected_actions)
        optimal_action = actions[max_idx]
        return optimal_action

    def target_soft_update(self, net, target_net, soft_tau):
        """ Soft update the target net """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        """ delayed update the target net """
        if self.step_cnt % update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data
                )

        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)
        torch.save(self.target_qnet1.state_dict(), path)
        torch.save(self.target_qnet2.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet1.load_state_dict(torch.load(path))
        self.target_qnet2.load_state_dict(torch.load(path))
        self.qnet.eval()
        self.target_qnet1.eval()
        self.target_qnet2.eval()
