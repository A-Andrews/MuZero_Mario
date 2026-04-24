"""MuZero CNN networks (representation + dynamics + prediction).

Architecture (matches the MuZero-Atari paper, adapted for 96x96x4 input with 4
stride-2 downsample stages -> 6x6x256 hidden state).

All three networks are bundled in a single `MuZeroNet` nn.Module:

  representation : (B, 4, 96, 96)          -> (B, 256, 6, 6)
  dynamics       : (B, 256, 6, 6), a:(B,)  -> (B, 256, 6, 6), reward_logits:(B, R)
  prediction     : (B, 256, 6, 6)          -> policy_logits:(B, A), value_logits:(B, V)

Hidden state is min-max normalised per-sample to [0, 1] after representation
and after each dynamics step. Reward and value are emitted as categorical-
support logits; `src.muzero.transforms` converts them to scalars via
signed-parabolic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


def _min_max_normalise(h: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max norm of a (B, C, H, W) tensor to [0, 1].

    Min/max are taken over all C*H*W activations of each sample, matching
    MuZero-Atari's hidden-state normalisation.
    """
    flat = h.view(h.size(0), -1)
    _min = flat.min(dim=1, keepdim=True)[0]
    _max = flat.max(dim=1, keepdim=True)[0]
    norm = (flat - _min) / (_max - _min + 1e-8)
    return norm.view_as(h)


class RepresentationNet(nn.Module):
    """(B, 4, 96, 96) -> (B, 256, 6, 6)."""

    def __init__(self, in_channels=4, hidden_channels=256, n_blocks=(2, 3, 3, 3)):
        super().__init__()
        ch1 = hidden_channels // 2  # 128
        ch2 = hidden_channels       # 256
        # Stage 1: 96 -> 48, 4 -> 128
        self.down1 = _conv3x3(in_channels, ch1, stride=2)
        self.bn1 = nn.BatchNorm2d(ch1)
        self.res1 = nn.Sequential(*[ResBlock(ch1) for _ in range(n_blocks[0])])
        # Stage 2: 48 -> 24, 128 -> 256
        self.down2 = _conv3x3(ch1, ch2, stride=2)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.res2 = nn.Sequential(*[ResBlock(ch2) for _ in range(n_blocks[1])])
        # Stage 3: 24 -> 12 via avgpool
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res3 = nn.Sequential(*[ResBlock(ch2) for _ in range(n_blocks[2])])
        # Stage 4: 12 -> 6 via avgpool
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.res4 = nn.Sequential(*[ResBlock(ch2) for _ in range(n_blocks[3])])

    def forward(self, x):
        x = F.relu(self.bn1(self.down1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.down2(x)))
        x = self.res2(x)
        x = self.pool3(x)
        x = self.res3(x)
        x = self.pool4(x)
        x = self.res4(x)
        return _min_max_normalise(x)


class DynamicsNet(nn.Module):
    """(h, action) -> (h', reward_logits)."""

    def __init__(
        self,
        hidden_channels=256,
        num_actions=12,
        n_blocks=15,
        spatial=6,
        reward_support_size=61,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.spatial = spatial
        self.conv_in = _conv3x3(hidden_channels + num_actions, hidden_channels)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(n_blocks)])
        # Reward head
        self.reward_conv = nn.Conv2d(hidden_channels, 16, kernel_size=1)
        self.reward_bn = nn.BatchNorm2d(16)
        self.reward_fc1 = nn.Linear(16 * spatial * spatial, 128)
        self.reward_fc2 = nn.Linear(128, reward_support_size)

    def forward(self, h, action_idx):
        B = h.size(0)
        # Broadcast one-hot action onto the spatial grid.
        action_one_hot = F.one_hot(action_idx, num_classes=self.num_actions).float()
        action_plane = action_one_hot.view(B, self.num_actions, 1, 1).expand(
            B, self.num_actions, self.spatial, self.spatial
        )
        x = torch.cat([h, action_plane], dim=1)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)
        h_next = _min_max_normalise(x)
        r = F.relu(self.reward_bn(self.reward_conv(h_next)))
        r = r.flatten(1)
        r = F.relu(self.reward_fc1(r))
        reward_logits = self.reward_fc2(r)
        return h_next, reward_logits


class PredictionNet(nn.Module):
    """h -> (policy_logits, value_logits)."""

    def __init__(
        self,
        hidden_channels=256,
        num_actions=12,
        n_blocks=2,
        spatial=6,
        value_support_size=601,
    ):
        super().__init__()
        self.spatial = spatial
        self.blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(n_blocks)])
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * spatial * spatial, num_actions)
        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(spatial * spatial, 256)
        self.value_fc2 = nn.Linear(256, value_support_size)

    def forward(self, h):
        x = self.blocks(h)
        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        policy_logits = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_fc2(v)
        return policy_logits, value_logits


class MuZeroNet(nn.Module):
    """Bundles representation + dynamics + prediction. Emits support logits."""

    def __init__(
        self,
        input_channels=4,
        input_spatial=96,
        hidden_channels=256,
        hidden_spatial=6,
        num_actions=12,
        value_support=(-300.0, 300.0, 601),
        reward_support=(-30.0, 30.0, 61),
        rep_blocks=(2, 3, 3, 3),
        dyn_blocks=15,
        pred_blocks=2,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_channels = hidden_channels
        self.hidden_spatial = hidden_spatial
        self.input_spatial = input_spatial

        self.value_support_min, self.value_support_max, self.value_support_size = value_support
        self.reward_support_min, self.reward_support_max, self.reward_support_size = reward_support

        self.representation = RepresentationNet(
            in_channels=input_channels,
            hidden_channels=hidden_channels,
            n_blocks=rep_blocks,
        )
        self.dynamics = DynamicsNet(
            hidden_channels=hidden_channels,
            num_actions=num_actions,
            n_blocks=dyn_blocks,
            spatial=hidden_spatial,
            reward_support_size=self.reward_support_size,
        )
        self.prediction = PredictionNet(
            hidden_channels=hidden_channels,
            num_actions=num_actions,
            n_blocks=pred_blocks,
            spatial=hidden_spatial,
            value_support_size=self.value_support_size,
        )

    # -- training-side unroll ----------------------------------------------

    def initial_step(self, obs):
        h = self.representation(obs)
        policy_logits, value_logits = self.prediction(h)
        return h, policy_logits, value_logits

    def recurrent_step(self, h, action_idx):
        h_next, reward_logits = self.dynamics(h, action_idx)
        policy_logits, value_logits = self.prediction(h_next)
        return h_next, reward_logits, policy_logits, value_logits

    # -- inference-side (MCTS) ---------------------------------------------

    @torch.no_grad()
    def initial_inference(self, obs):
        """obs: (B, C, H, W) float tensor. Returns numpy/torch objects for MCTS."""
        h = self.representation(obs)
        policy_logits, value_logits = self.prediction(h)
        value = self._value_scalar(value_logits)
        return h, policy_logits, value

    @torch.no_grad()
    def recurrent_inference(self, h, action_idx):
        h_next, reward_logits = self.dynamics(h, action_idx)
        policy_logits, value_logits = self.prediction(h_next)
        reward = self._reward_scalar(reward_logits)
        value = self._value_scalar(value_logits)
        return h_next, reward, policy_logits, value

    # -- support <-> scalar ------------------------------------------------

    def _value_scalar(self, logits):
        from src.muzero.transforms import support_to_scalar
        return support_to_scalar(
            logits, self.value_support_min, self.value_support_max, self.value_support_size
        )

    def _reward_scalar(self, logits):
        from src.muzero.transforms import support_to_scalar
        return support_to_scalar(
            logits, self.reward_support_min, self.reward_support_max, self.reward_support_size
        )
