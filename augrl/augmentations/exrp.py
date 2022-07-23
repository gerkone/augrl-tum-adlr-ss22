import gym
from typing import Tuple, Dict
import numpy as np
from math import prod
import torch
import torch.nn as nn
from tqdm import tqdm


class ExplicitRewardPredictor:
    def __init__(
        self,
        predictor_net: nn.Module,
        device,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size

        self.device = device
        self.predictor_net = predictor_net.to(device)

        self.optimizer = torch.optim.Adam(
            self.predictor_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.95
        )

        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def from_env(cls, env: gym.Env,
                 device,
                 lr: float = 1e-4,
                 batch_size: int = 32,):
        state_embed = _dense_embed(env.observation_space.shape)
        action_embed = _dense_embed(env.action_space.shape)
        predictor = PredictorNetwork(state_embed, action_embed)
        return cls(predictor, device, lr, batch_size)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> float:
        return self.predictor_net(state.to(self.device), action.to(self.device)).detach().cpu()

    def prefer(self, segments: Dict) -> int:
        "Returns 0 if left trajectory is preferred (ie predicted left reward is larger than predicted right reward)."
        return (self.predictor_net.cumulate_on_segment(segments["obs_left"].to(self.device), segments["act_left"].to(self.device)) < self.predictor_net.cumulate_on_segment(segments["obs_right"].to(self.device), segments["act_right"].to(self.device))).int().detach().cpu()

    def train(self, segments: Dict, epochs: int = 10, show_pregress: bool = True):
        """Train the predictor on passed segment preferences, using cross entropy loss.

        This is based on the idea of estimating score functions from pairwise differences. The loss depends
        on wether the preference is assigned to the segment with the higher reward. This pushes segments 
        with preference to have a higher reward.

        The segments dict should have left/right observations and activities as lists of segments, and
        for each pair of segments a preference.
        """
        if (
            not len(segments["obs_left"])
            == len(segments["obs_right"])
            == len(segments["act_left"])
            == len(segments["act_right"])
            == len(segments["preferences"])
        ):
            raise ValueError(
                "Passed segments have mismatching lengths:\n"
                + "obs left: {}, obs_right: {}, act_left: {}, act_right: {}, preferences: {}".format(
                    len(segments["obs_left"]),
                    len(segments["obs_right"]),
                    len(segments["act_left"]),
                    len(segments["act_right"]),
                    len(segments["preferences"]),
                )
            )
        if show_pregress:
            pbar = tqdm(range(epochs))
            pbar.clear()
            pbar.set_description("Epoch {:2} Loss {:5.3f}".format(1, 0))
        else:
            pbar = range(epochs)

        losses = []
        for e in pbar:
            observation_left_dl = iter(torch.utils.data.DataLoader(
                segments["obs_left"], batch_size=self.batch_size, shuffle=False
            ))
            observation_right_dl = iter(torch.utils.data.DataLoader(
                segments["obs_right"], batch_size=self.batch_size, shuffle=False
            ))
            action_left_dl = iter(torch.utils.data.DataLoader(
                segments["act_left"], batch_size=self.batch_size, shuffle=False
            ))
            action_right_dl = iter(torch.utils.data.DataLoader(
                segments["act_right"], batch_size=self.batch_size, shuffle=False
            ))
            preference_dl = iter(torch.utils.data.DataLoader(
                segments["preferences"], batch_size=self.batch_size, shuffle=False
            ))
            while True:
                try:
                    obs_left_batch = next(observation_left_dl)
                    act_left_batch = next(
                        action_left_dl
                    )
                    obs_right_batch = next(observation_right_dl)
                    act_right_batch = next(
                        action_right_dl
                    )
                    pref_batch = next(preference_dl)
                    # batched obs and act have shape (batch, segment length, ...)
                    self.optimizer.zero_grad()
                    # get total segment reward
                    cumulated_reward_left = self.predictor_net.cumulate_on_segment(
                        obs_left_batch.to(self.device), act_left_batch.to(self.device))
                    cumulated_reward_right = self.predictor_net.cumulate_on_segment(
                        obs_right_batch.to(self.device), act_right_batch.to(self.device))
                    # should have shape (batch, 2)
                    reward_comparison = torch.stack(
                        [cumulated_reward_left, cumulated_reward_right], axis=1
                    )
                    loss = self.loss_fn(reward_comparison, pref_batch.to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss)
                except StopIteration:
                    break
            self.scheduler.step()
            if show_pregress:
                pbar.set_description(
                    "Epoch {:2} Loss {:5.3f}".format(e + 1, sum(losses) / len(losses))
                )
        if show_pregress:
            pbar.close()


class PredictorNetwork(nn.Module):
    def __init__(self, state_embed: nn.Module, action_embed: nn.Module) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        # state-action contribution control
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.predictor = nn.Sequential(
            nn.Linear(100, 256), nn.LeakyReLU(True), nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if len(state.shape) == 1:
            action = state.reshape(state.shape[0], 1)
        if len(action.shape) == 1:
            action = action.reshape(action.shape[0], 1)
        s_em = self.state_embed(state) * self.alpha
        a_em = self.action_embed(action) * (1 - self.alpha)
        x = torch.add(s_em, a_em)
        x = nn.LeakyReLU(True)(x)
        return self.predictor(x)

    def cumulate_on_segment(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack([
            self(obs.reshape(obs.shape[0], *obs.shape[2:]), act.reshape(act.shape[0], *act.shape[2:])).squeeze()
            for obs, act in zip(
                torch.split(obs, 1, dim=1),
                torch.split(act, 1, dim=1),
            )], axis=-1), axis=-1)


def _dense_embed(s: Tuple):
    return nn.Sequential(
        nn.Linear(prod(s), 100),
        nn.LeakyReLU(True),
        nn.Linear(100, 100),
    )


def get_segments(obs_spin: torch.Tensor, act_spin: torch.Tensor, obs_no_spin: torch.Tensor, act_no_spin: torch.Tensor) -> Dict:
    segments = {}
    segments["obs_left"] = []
    segments["obs_right"] = []
    segments["act_left"] = []
    segments["act_right"] = []
    segments["preferences"] = []
    for obs_1, act_1 in zip(torch.split(obs_spin, 1, dim=0), torch.split(act_spin, 1, dim=0)):
        for obs_2, act_2 in zip(torch.split(obs_no_spin, 1, dim=0), torch.split(act_no_spin, 1, dim=0)):
            obs_1 = obs_1.squeeze()
            obs_2 = obs_2.squeeze()
            act_1 = act_1.squeeze()
            act_2 = act_2.squeeze()
            # not sure if this is needed
            if np.random.random() > 0.5:
                segments["obs_left"].append(obs_1)
                segments["obs_right"].append(obs_2)
                segments["act_left"].append(act_1)
                segments["act_right"].append(act_2)
                # left is the one with the spin
                segments["preferences"].append(0)
            else:
                segments["obs_left"].append(obs_2)
                segments["obs_right"].append(obs_1)
                segments["act_left"].append(act_2)
                segments["act_right"].append(act_1)
                # right is the one with the spin
                segments["preferences"].append(1)
    segments["obs_left"] = torch.stack(segments["obs_left"], dim=0)
    segments["obs_right"] = torch.stack(segments["obs_right"], dim=0)
    segments["act_left"] = torch.stack(segments["act_left"], dim=0)
    segments["act_right"] = torch.stack(segments["act_right"], dim=0)
    segments["preferences"] = torch.tensor(segments["preferences"])
    assert segments["obs_left"].shape == segments["obs_right"].shape
    assert segments["act_left"].shape == segments["act_right"].shape
    return segments
