from __future__ import annotations

import io
import itertools
import pathlib
import random
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F


def dict_deep_iterate(d: dict, keys: list):
    """
    Iterate over a nested dictionary given a list of keys
    """
    for k in keys:
        d = d[k]
    return d


SelfTADPO = TypeVar("SelTADPO", bound="TADPO")


class TADPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param teacher_policy: The teacher policy model to use (Inherits Base Policy)
    :param teacher_policy_observation_key: The keys used to extract teacher policy observation from the environment's info dict
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param teacher_n_steps: The size of the teacher rollout buffer
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param teacher_training_prob: Probability of training on teacher data
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param clip_range_teacher: Clipping parameter for the teacher policy (similar to clip_range)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param teacher_ent_coef: Entropy coefficient for the loss calculation
    :param tad_coef: Policy loss coefficient for the teacher action distillation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param deterministic_teacher_rollout: Whether to use deterministic teacher actions
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    rollout_buffer: RolloutBuffer
    teacher_rollout_buffer: RolloutBuffer

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        teacher_policy: BasePolicy = None,
        teacher_policy_observation_key: list[str] | None = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        teacher_n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        teacher_training_prob: int = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        clip_range_teacher: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        teacher_ent_coef: float = 0.001,
        tad_coef: float = 0.4,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        deterministic_teacher_rollout: bool = False,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = {},
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.teacher_training_prob = teacher_training_prob
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.clip_range_teacher = clip_range_teacher
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.teacher_n_steps = teacher_n_steps
        self.teacher_policy = teacher_policy
        self.teacher_policy_observation_key = teacher_policy_observation_key
        self.teacher_policy_observation_space = (
            teacher_policy.observation_space if self.teacher_policy else None
        )
        self.deterministic_teacher_rollout = deterministic_teacher_rollout

        self.teacher_ent_coef = teacher_ent_coef
        self.tad_coef = tad_coef

        self._last_teacher_obs = None
        self._last_rollout_actor = None
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        self.clip_range_teacher = get_schedule_fn(self.clip_range_teacher)
        self.teacher_training_prob = get_schedule_fn(self.teacher_training_prob)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        if self.teacher_policy:
            self.teacher_policy.to(self.device)
            self.teacher_policy.set_training_mode(False)

        if self.teacher_n_steps > 0 and self.teacher_policy:
            if isinstance(self.observation_space, spaces.Dict):
                if not isinstance(self.teacher_policy_observation_space, spaces.Dict):
                    raise ValueError(
                        "If main policy uses dict observation space, "
                        "the teacher policy also needs to use a Dict observation space"
                    )
                self.teacher_rollout_buffer = DictRolloutBuffer(
                    self.teacher_n_steps,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    n_envs=self.n_envs,
                    **self.rollout_buffer_kwargs,
                )
            else:
                if isinstance(self.teacher_policy_observation_space, spaces.Dict):
                    raise ValueError(
                        "If main policy does not use dict observation space, "
                        "the teacher policy also cannot to use a Dict observation space"
                    )
                self.teacher_rollout_buffer = RolloutBuffer(
                    self.teacher_n_steps,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    n_envs=self.n_envs,
                    **self.rollout_buffer_kwargs,
                )
        else:
            self.teacher_rollout_buffer = None

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        if not self.teacher_policy:
            raise ValueError("Cannot learn without a teacher policy")
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        self._last_teacher_obs = self._get_teacher_obs_from_info(self.env.reset_infos)
        return total_timesteps, callback

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_teacher = self.clip_range_teacher(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        teacher_training_prob = self.teacher_training_prob(
            self._current_progress_remaining
        )

        # entropy loss
        entropy_ls = []
        # student policy gradient loss
        policy_gradient_ls = []
        # value function loss
        vf_ls = []
        # student clip fractions
        clip_fractions = []
        # teacher clip fractions
        teacher_clip_fractions = []
        # teacher action distillation loss
        tad_ls = []

        # train teacher fraction
        train_batch_is_teacher = []

        continue_training = True
        trained_using_teacher = False
        trained_using_student = False

        def get_next_batch():
            if self.teacher_rollout_buffer:
                is_teacher = True
                teacher_rollout_generator = zip(
                    itertools.repeat(is_teacher),
                    self.teacher_rollout_buffer.get(self.batch_size),
                )
            else:
                teacher_rollout_generator = None
            if self.rollout_buffer:
                is_teacher = False
                student_rollout_generator = zip(
                    itertools.repeat(is_teacher),
                    self.rollout_buffer.get(self.batch_size),
                )
            else:
                student_rollout_generator = None

            use_student_only = (teacher_rollout_generator is None) and (
                student_rollout_generator is not None
            )
            use_student_only |= teacher_training_prob < 1e-8

            use_teacher_only = (teacher_rollout_generator is not None) and (
                student_rollout_generator is None
            )
            use_teacher_only |= teacher_training_prob > 1 - 1e-8

            if use_student_only:
                for rollout_data in student_rollout_generator:
                    yield rollout_data
            elif use_teacher_only:
                for rollout_data in teacher_rollout_generator:
                    yield rollout_data
            else:
                while True:
                    # Decide which generator to use based on the given probability
                    if random.random() < teacher_training_prob:
                        try:
                            # Yield from teacher if probability favors it
                            yield next(teacher_rollout_generator)
                        except StopIteration:
                            # If the teacher generator is exhausted, yield from the student
                            yield from student_rollout_generator
                            break
                    else:
                        try:
                            # Yield from student if probability favors it
                            yield next(student_rollout_generator)
                        except StopIteration:
                            # If the student generator is exhausted, stop
                            break

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for is_teacher, rollout_data in get_next_batch():
                train_batch_is_teacher.append(is_teacher)
                if is_teacher:
                    # Do training on teacher data
                    trained_using_teacher = True
                    # Do a complete pass on the rollout buffer
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    # Try to reduce number of forward passes on network
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    teacher_log_prob = rollout_data.old_log_prob

                    student_advantage = rollout_data.returns - values
                    # normalize mean by advantage of teacher
                    if len(student_advantage) > 1:
                        student_advantage = (
                            student_advantage - rollout_data.advantages.mean()
                        ) / (torch.std(student_advantage) + 1e-8)

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_ls.append(entropy_loss.item())

                    # ratio between teacher and new policy
                    teacher_action_student_ratio = torch.exp(
                        log_prob - teacher_log_prob
                    )
                    teacher_action_distillation_loss = student_advantage * torch.clamp(
                        teacher_action_student_ratio,
                        None,
                        1 + clip_range_teacher,
                    )
                    teacher_action_distillation_loss = -torch.clamp(
                        teacher_action_distillation_loss, 0
                    ).mean()
                    tad_ls.append(teacher_action_distillation_loss.item())

                    teacher_clip_fraction = torch.mean(
                        ((teacher_action_student_ratio - 1) > clip_range).float()
                    ).item()
                    teacher_clip_fractions.append(teacher_clip_fraction)

                    loss = 0
                    coefs_and_losses = [
                        (
                            self.teacher_ent_coef,
                            entropy_loss,
                        ),
                        (
                            self.tad_coef,
                            teacher_action_distillation_loss,
                        ),
                    ]

                    for c, l in coefs_and_losses:
                        if c > 1e-8:
                            # we do not add the values if c is 0
                            loss += c * l

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    self.policy.optimizer.step()
                else:
                    trained_using_student = True
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    policy_gradient_ls.append(policy_loss.item())
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    vf_ls.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_ls.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                    )

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = (
                            torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                            .cpu()
                            .numpy()
                        )
                        approx_kl_divs.append(approx_kl_div)

                    if (
                        self.target_kl is not None
                        and approx_kl_div > 1.5 * self.target_kl
                    ):
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                            )
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        if trained_using_student:
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_ls))
        self.logger.record("train/value_loss", np.mean(vf_ls))
        self.logger.record(
            "train/teacher_batch_fraction", np.mean(train_batch_is_teacher)
        )
        self.logger.record("train/loss", loss.item())
        if trained_using_student:
            self.logger.record(
                "train/policy_gradient_loss", np.mean(policy_gradient_ls)
            )
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/explained_variance", explained_var)
        if trained_using_teacher:
            self.logger.record(
                "train/tad_loss",
                np.mean(tad_ls),
            )
            self.logger.record(
                "train/teacher_clip_fraction", np.mean(teacher_clip_fractions)
            )

        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfTADPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTADPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        if self.teacher_rollout_buffer.full:
            # Give access to local variables
            callback.update_locals(locals())
        else:
            self._collect_teacher_rollout(
                env, callback, self.teacher_rollout_buffer, self.teacher_n_steps
            )
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def _collect_teacher_rollout(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ):
        if self.teacher_policy is None:
            raise ValueError(
                "Teacher model is necessary if rollout uses teacher action"
            )

        # We do not need to reset the actor in envs
        # because the advantage is not propagated
        # towards the inital values in the rollout buffer

        # if (
        #     self._last_rollout_actor != "teacher"
        #     and self._last_rollout_actor is not None
        # ):
        #     self._reset_envs_and_terminate_existing_episodes()
        self._last_rollout_actor = "teacher"
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                if self.teacher_policy_observation_key is None:
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    teacher_obs_tensor = obs_tensor
                else:
                    teacher_obs_tensor = obs_as_tensor(
                        self._last_teacher_obs, self.device
                    )
                actions, teacher_values, teacher_log_probs = self.teacher_policy(
                    teacher_obs_tensor, deterministic=self.deterministic_teacher_rollout
                )

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.teacher_policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.teacher_policy.unscale_action(
                        clipped_actions
                    )
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                teacher_values,
                teacher_log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_teacher_obs = self._get_teacher_obs_from_info(infos)
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            teacher_values = self.teacher_policy.predict_values(
                obs_as_tensor(self._last_teacher_obs, self.device)
            )  # type: ignore[arg-type]

        if rollout_buffer.buffer_size > 1e3:
            print("Computing teacher rollout returns and advantage")
        rollout_buffer.compute_returns_and_advantage(
            last_values=teacher_values, dones=dones
        )

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _get_teacher_obs_from_info(self, infos: list[dict[str, Any]]):
        if not self.teacher_policy_observation_space and self.teacher_policy:
            self.teacher_policy_observation_space = (
                self.teacher_policy.observation_space
            )
        obs_space = self.teacher_policy_observation_space
        obs_list = [
            dict_deep_iterate(i, self.teacher_policy_observation_key) for i in infos
        ]
        obs = {}
        if isinstance(obs_space, spaces.Dict):
            for k in obs_space.keys():
                obs[k] = np.array([o[k] for o in obs_list])
        return obs

    def _save_buffer(
        self, buffer: RolloutBuffer, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ):
        save_to_pkl(path, buffer, self.verbose)

    def save_rollout_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        assert self.rollout_buffer is not None, "The rollout buffer is not defined"
        self._save_buffer(self.rollout_buffer, path)

    def save_teacher_rollout_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        assert (
            self.teacher_rollout_buffer is not None
        ), "The rollout buffer is not defined"
        self._save_buffer(self.teacher_rollout_buffer, path)

    def load_rollout_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        self.rollout_buffer = load_from_pkl(path, self.verbose)
        # assert isinstance(
        #     self.rollout_buffer, RolloutBuffer
        # ), "The rollout buffer must inherit from RolloutBuffer class"

        # Update saved rollout buffer device to match current setting, see GH#1561
        self.rollout_buffer.device = self.device

    def load_teacher_rollout_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
    ) -> None:
        self.teacher_rollout_buffer = load_from_pkl(path, self.verbose)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "teacher_policy",
            "teacher_rollout_buffer",
        ]

    def _reset_envs_and_terminate_existing_episodes(self):
        obs = self.env.reset()
        reset_infos = self.env.reset_infos
        self._last_episode_starts = np.ones_like(self._last_episode_starts)
        self._last_obs = obs
        self._last_teacher_obs = self._get_teacher_obs_from_info(reset_infos)
