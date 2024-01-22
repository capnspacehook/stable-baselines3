from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch as th

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs

class VecBatcher(VecEnvWrapper):
    def __init__(self, venv: VecEnv, get_actions: Callable[[VecEnvObs], Tuple[np.ndarray, Tuple]], pre_process_actions: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__(venv)

        self.get_actions = get_actions
        self.pre_process_actions = pre_process_actions
        self.num_steps = 0

        self._last_batch_obs = None
        self._seen_steps = set()
        self._next_envs = []
        self._next_env_steps = {env_id: 0 for env_id in range(venv.num_envs)}
        self._training_data = {}
        self._results = {}

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self._last_batch_obs = obs
        return obs

    def full_step(self) -> Tuple[VecEnvStepReturn, Tuple]:
        while True:
            actions, output = self.get_actions(self._last_batch_obs)
            zipped_output = tuple(zip(*output))

            # Store outputs so they can be grouped with step returns once all envs finish a step
            if len(actions) == self.num_envs:
                self._seen_steps.add(self.num_steps)
                self._training_data[self.num_steps] = [None] * self.num_envs
                for env_id, action in enumerate(actions):
                    self._training_data[self.num_steps][env_id] = (action, *zipped_output[env_id])
            else:
                for i, env_id in enumerate(self._next_envs):
                    step = self._next_env_steps[env_id]
                    if step not in self._training_data:
                        self._seen_steps.add(step)
                        self._training_data[step] = [None] * self.num_envs

                    self._training_data[step][env_id] = (actions[i], *zipped_output[i])

            processed_actions = actions
            if self.pre_process_actions is not None:
                processed_actions = self.pre_process_actions(actions)

            new_obs, rewards, dones, infos = self.venv.step(processed_actions)

            self.num_steps += len(infos)
            # If the VecEnv returned results for all envs, just return them
            if len(infos) == self.num_envs:
                # Observations will be flattened if results from all envs
                # was returned
                self._last_batch_obs = new_obs
                return (new_obs, rewards, dones, infos), (actions, *output)
            
            self._last_batch_obs = _flatten_obs(new_obs, self.venv.observation_space)

            # Store the returned results, there may not be enough results
            # to return step results from every env yet
            seen_envs = []
            for i, info in enumerate(infos):
                step = info["step"]
                env_id = info["env_id"]
                if step not in self._results:
                    self._results[step] = [None] * self.num_envs

                seen_envs.append(env_id)
                self._next_env_steps[env_id] = step + 1
                self._results[step][env_id] = (new_obs[i], rewards[i], dones[i], info)

            self._next_envs = seen_envs

            # If the smallest step we've seen has results from all envs,
            # process and return the results
            min_step = min(self._seen_steps)
            if None not in self._results[min_step]:
                outputs = list(zip(*self._training_data[min_step]))
                for i, output in enumerate(outputs):
                    if isinstance(output[0], th.Tensor):
                        outputs[i] = th.stack(output)
                    else:
                        outputs[i] = np.stack(output)

                new_obs, rewards, dones, infos = zip(*self._results[min_step])

                self._seen_steps.remove(min_step)
                del self._training_data[min_step]
                del self._results[min_step]

                return (
                    _flatten_obs(new_obs, self.venv.observation_space),
                    np.stack(rewards),
                    np.stack(dones),
                    infos,
                ), outputs

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        raise NotImplementedError
