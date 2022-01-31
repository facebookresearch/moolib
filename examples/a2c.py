# Copyright (c) Facebook, Inc. and its affiliates.

import csv
import os
import time

import gym
import torch
from torch import nn
from torch.nn import functional as F

import moolib


ADDRESS = "127.0.0.1:5541"

TOTAL_STEPS = 50000

BATCH_SIZE = 2
ROLLOUT_LENGTH = 64

DISCOUNT = 0.99
LR = 1e-3
BASELINE_COST = 0.005
ENTROPY_COST = 0.0006
ADAM_BETAS = (0.0, 0.99)
ADAM_EPSILON = 3e-7

USE_LSTM = True


def log_to_file(_state={}, **fields):  # noqa: B006
    """Incrementally write logs.tsv in cwd."""
    if "writer" not in _state:
        path = "logs.tsv"
        writeheader = not os.path.exists(path)
        fieldnames = list(fields.keys())

        _state["file"] = open(path, "a", buffering=1)  # Line buffering.
        _state["writer"] = csv.DictWriter(_state["file"], fieldnames, delimiter="\t")
        if writeheader:
            _state["writer"].writeheader()
            print("Writing logs to", path)
        else:
            print("Appending logs to", path)

    writer = _state["writer"]
    if writer is not None:
        writer.writerow(fields)


class Model(nn.Module):
    def __init__(self, num_actions=2):
        super(Model, self).__init__()

        self.linear0 = nn.Linear(4, 128)
        self.linear1 = nn.Linear(128, 32)

        core_output_size = self.linear1.out_features

        if USE_LSTM:
            self.core = nn.LSTMCell(core_output_size, 32)
            core_output_size = 32

        self.policy = nn.Linear(32, num_actions)
        self.baseline = nn.Linear(32, 1)

    def initial_state(self, batch_size=1):
        if not USE_LSTM:
            return tuple()
        return tuple(torch.zeros(batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, observation, done, core_state, unroll=False):
        if not unroll:
            # [B, ...] -> [T=1, B, ...].
            observation = observation.unsqueeze(0)

        T, B, *_ = observation.shape
        x = observation.reshape(T * B, -1)

        x = torch.tanh(self.linear0(x))
        x = torch.tanh(self.linear1(x))

        core_input = x

        if USE_LSTM:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            notdone.unsqueeze_(-1)  # [T, B, H=1] for broadcasting.

            for input_t, notdone_t in zip(core_input.unbind(), notdone.unbind()):
                core_state = tuple(notdone_t * t for t in core_state)
                output_t, core_state = self.core(input_t, core_state)
                core_state = (output_t, core_state)  # nn.LSTMCell is a bit weird.
                core_output_list.append(output_t)  # [[B, H], [B, H], ...].
            core_output = torch.cat(core_output_list)  # [T * B, H].
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        action = action.view(T, B)
        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)

        output = (action, policy_logits, baseline)
        if not unroll:
            for t in output:
                t.squeeze_(0)
        return output, core_state


def create_env():
    return gym.make("CartPole-v0")


def a2c_loss(state, action, reward, done, initial_core_state, model):
    # This time with gradients.
    (unused_actions, policy_logits, baseline), _ = model(
        state, done, initial_core_state, unroll=True
    )

    reward = reward[1:]
    done = done[1:]
    policy_logits = policy_logits[:-1]
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]
    action = action[:-1]

    T, B, *_ = reward.shape

    discount = ~done * DISCOUNT
    returns = torch.empty(T, B)

    acc = bootstrap_value.detach()
    for t in range(T - 1, -1, -1):
        acc = reward[t] + acc * discount[t]
        returns[t] = acc

    advantages = returns - baseline

    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)

    cross_entropy = torch.gather(log_policy, dim=-1, index=action[..., None])
    cross_entropy.squeeze_(-1)

    # Alternative:
    # cross_entropy = F.nll_loss(
    #     F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
    #     target=torch.flatten(actions, 0, 1),
    #     reduction="none",
    # )
    # cross_entropy = cross_entropy.view_as(returns)

    pg_loss = -torch.mean(cross_entropy * advantages.detach())
    baseline_loss = 0.5 * torch.mean(advantages**2)
    entropy_loss = torch.mean(policy * log_policy)

    return pg_loss, baseline_loss, entropy_loss


def train(total_steps=TOTAL_STEPS):
    # EnvPool runs a batch of environments in separate processes,
    # create_env is a user-defined function that returns a gym environment.
    envs = moolib.EnvPool(
        create_env, num_processes=1, batch_size=BATCH_SIZE, num_batches=1
    )

    model = Model()

    broker = moolib.Broker()
    broker.listen(ADDRESS)
    accumulator = moolib.Accumulator("foo", model.parameters(), model.buffers())
    accumulator.connect(ADDRESS)

    # Our optimizer.
    opt = torch.optim.Adam(
        model.parameters(), lr=LR, betas=ADAM_BETAS, eps=ADAM_EPSILON
    )

    states = []
    actions = []
    rewards = []
    dones = []
    episode_returns = []

    action_t = torch.zeros(BATCH_SIZE, dtype=torch.int64)
    episode_step_t = torch.zeros(BATCH_SIZE, dtype=torch.int64)
    episode_return_t = torch.zeros(BATCH_SIZE)

    core_state_t = model.initial_state(batch_size=BATCH_SIZE)
    core_states = []

    local_loss_computes = 0
    local_update_steps = 0

    while True:
        # update does some internal moolib book-keeping,
        # makes sure we're still connected, etc.
        broker.update()
        accumulator.update()

        if not accumulator.connected():
            # If we're not connected, sleep for a bit so we don't busy-wait
            print("Your training will commence shortly.")
            time.sleep(0.25)
            continue

        # TODO: What's the 0? (Subgroup?)
        obs_future = envs.step(0, action_t)
        obs = obs_future.result()

        episode_step_t += 1
        episode_return_t += obs["reward"]

        core_states.append(core_state_t)

        state_t = obs["state"].to(torch.float, copy=True)
        with torch.no_grad():
            (action_t, logits_t, _), core_state_t = model(
                state_t, obs["done"], core_state_t
            )

        episode_returns.append(episode_return_t.clone())
        episode_step_t *= ~obs["done"]
        episode_return_t *= ~obs["done"]

        states.append(state_t)
        actions.append(action_t)
        rewards.append(obs["reward"].clone())
        dones.append(obs["done"].clone())

        if accumulator.wants_state():
            # For sharing the initial optimizer state with new nodes.
            accumulator.set_state({"optimizer": opt.state_dict()})

        if accumulator.has_new_state():
            # This node wants an initial optimizer state.
            opt.load_state_dict(accumulator.state()["optimizer"])

        if accumulator.wants_gradients():
            if len(states) < ROLLOUT_LENGTH + 1:
                # No data to train on, so this node won't participate in the
                # gradient reduction this time. Gradients may still be reduced
                # from the other nodes.
                accumulator.skip_gradients()
            else:
                # Train step: We have data to train on. Calculate
                # a local gradient and initiate a reduction.
                state = torch.stack(states)
                action = torch.stack(actions)
                reward = torch.stack(rewards)
                done = torch.stack(dones)

                pg_loss, baseline_loss, entropy_loss = a2c_loss(
                    state, action, reward, done, core_states[0], model
                )

                mean_episode_return = float("nan")  # Mean of empty set.
                if done.any():
                    returns = torch.stack(episode_returns)
                    mean_episode_return = torch.mean(returns[done]).item()

                del states[:-1]
                del actions[:-1]
                del rewards[:-1]
                del dones[:-1]
                del episode_returns[:-1]
                del core_states[:-1]

                loss = (
                    pg_loss
                    + BASELINE_COST * baseline_loss
                    + ENTROPY_COST * entropy_loss
                )
                loss.backward()
                local_loss_computes += 1

                agent_steps = local_loss_computes * ROLLOUT_LENGTH * BATCH_SIZE

                if agent_steps >= total_steps:
                    print("Training ended after %i steps." % agent_steps)
                    return

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

                if local_loss_computes % 100 == 0:
                    mer = mean_episode_return
                    print(
                        "Local loss computation %d. (%d agent steps) "
                        "Mean episode return: %s (%d episode ends). "
                        "Local update steps: %d"
                        % (
                            local_loss_computes,
                            agent_steps,
                            None if mer != mer else "%f" % mer,  # NaN -> None.
                            torch.sum(done),
                            local_update_steps,
                        )
                    )
                log_to_file(
                    step=agent_steps,
                    mean_episode_return=mean_episode_return,
                    pg_loss=pg_loss.item(),
                    baseline_loss=baseline_loss.item(),
                    entropy_loss=entropy_loss.item(),
                    grad_norm=grad_norm.item(),
                )
                # Trigger an asynchronous gradient reduction.
                # has_gradients() will return true when the reduction is done.
                # We tell it the batch size so we can have virtual batch sizes
                # and communicate the number of batches consumed.
                accumulator.reduce_gradients(BATCH_SIZE)

        if accumulator.has_gradients():
            # moolib has reduced gradients, so we can step the optimizer
            opt.step()
            accumulator.zero_gradients()  # has_gradients() will return false after this
            local_update_steps += 1

        # While we are waiting for gradients, we can't do any more train
        # steps but we can still run inference to generate more data.
        # That will happen in the next iteration.


def main():
    train()


if __name__ == "__main__":
    main()
