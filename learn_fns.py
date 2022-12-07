import torch
import torch.nn.functional as F


def Q(q_net, states, actions):
    return q_net(states).gather(1, actions)


def learn_dqn_with_target(qnet_primary, qnet_target, states, actions, rewards, next_states, dones, gamma):
    """
    :param qnet_primary: primary network
    :param qnet_target: target network
    :param state: state at time t
    :param action: action at time t
    :param rewards: rewards at time t
    :param next_states: states at time t+1
    :param dones: 
    :param gamma: discount factor
    :param optim: optimizer used by the agent
    :return:
    mse_loss: mean squared error loss of the current batch
    """
    with torch.no_grad():
        maxQ = qnet_target(next_states).max(1)[0].unsqueeze(1)

    y_DQN = rewards + gamma * maxQ * (1 - dones)
    return F.mse_loss(
        y_DQN,
        Q(qnet_primary, states, actions)
    )


def learn_ddqn_with_target(qnet_primary, qnet_target, states, actions, rewards, next_states, dones, gamma):
    # with torch.no_grad():
    #     # max Q_target(s', a)
    #     #  a
    #     q_next_state = self.qnetwork_target(next_states).cpu().max(1)[0].unsqueeze(1)
    # y_DQN = rewards + self.gamma * q_next_state * (1.0 - dones)
    with torch.no_grad():
        #
        # 
        max_actions = qnet_primary(next_states).max(1)[1].unsqueeze(1)
        q_next_state = Q(qnet_target, next_states, max_actions)
        y_DDQN = rewards + gamma * q_next_state * (1.0 - dones)
    return F.mse_loss(
        y_DDQN,
        Q(qnet_primary, states, actions)
    )
