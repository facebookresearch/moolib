# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gym

from . import atari_preprocessing


def create_env(flags):
    env = gym.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
        flags.env.name,
        obs_type="grayscale",  # "ram", "rgb", or "grayscale".
        frameskip=1,  # Action repeats. Done in wrapper b/c of noops.
        repeat_action_probability=flags.env.repeat_action_probability,  # Sticky actions.
        full_action_space=True,  # Use all actions.
        render_mode=None,  # None, "human", or "rgb_array".
    )

    # Using wrapper from seed_rl in order to do random no-ops _before_ frameskipping.
    # gym.wrappers.AtariPreprocessing doesn't play well with the -v5 versions of the game.
    env = atari_preprocessing.AtariPreprocessing(
        env,
        frame_skip=flags.env.num_action_repeats,
        terminal_on_life_loss=False,
        screen_size=84,
        max_random_noops=flags.env.noop_max,  # Max no-ops to apply at the beginning.
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env
