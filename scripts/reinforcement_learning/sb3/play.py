# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)

# Moji argument
parser.add_argument("--log-tilt", type=int, default=0, help="Number of timesteps to log robot tilt (0 = disable).")
parser.add_argument("--log_io", type=int, default=0, help="Number of io samples to log (0 = disable).")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

from isaaclab.utils.math import quat_apply

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", train_task_name)
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    vec_norm_path = checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    timestep = 0
    run_id = 0

    time_log = []
    tilt_log = []

    input_log = []
    output_log = []

    logged_samples = 0
    done_sampling = False

    robot = env.unwrapped.scene["robot"]

    # folder relative to the script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tilt_log_folder = os.path.join(base_dir, "../../../tilt_logging")
    os.makedirs(tilt_log_folder, exist_ok=True)

    io_log_folder = os.path.join(base_dir, "../../../io_logging")
    os.makedirs(io_log_folder, exist_ok=True)


    def compute_tilt():
        root_quat = robot.data.root_quat_w  # [N, 4]
        batch_size = root_quat.shape[0]

        world_up = torch.tensor([0, 0, 1], device=root_quat.device, dtype=root_quat.dtype).repeat(batch_size, 1)
        up_vector = quat_apply(root_quat, world_up)

        # Use atan2 to get signed tilt around XZ plane (for example roll vs. yaw)
        # sign from projection on x-axis
        tilt_signed = torch.atan2(up_vector[:, 0], up_vector[:, 2])  # roll tilt
        # Or use y if you want pitch tilt: atan2(up_vector[:, 1], up_vector[:, 2])

        return tilt_signed

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # log tilt if within run count
        if run_id < args_cli.log_tilt:
            tilt_deg = compute_tilt()[0].item() * 180.0 / 3.14159
            sim_time = timestep * dt
            time_log.append(sim_time)
            tilt_log.append(tilt_deg)

        if args_cli.log_io > 0 and logged_samples < args_cli.log_io and not done_sampling:
            input_log.append(obs[0])
            output_log.append(actions[0])
            logged_samples += 1
            print(f"\rSample {logged_samples}", end="", flush=True)

            # print("Obs[0]: ", obs[0])
            # print("actions[0]: ", actions[0])


        # check loop reset (600 steps = 1 loop)
        # if timestep >= (env.unwrapped.episode_length_s * (1 / dt)) / env.unwrapped.decimation:
        if timestep >= env.unwrapped.max_episode_length:
            run_id += 1
            if run_id <= args_cli.log_tilt:
                tilt_log_filename = os.path.join(tilt_log_folder, f"tilt_log_run{run_id}.npz")
                np.savez(tilt_log_filename, time=np.array(time_log), tilt=np.array(tilt_log))
                print(f"Saved tilt log to {tilt_log_filename}")
                # reset logs for next loop
            time_log, tilt_log = [], []
            timestep = 0

            # stop logging if finished required runs
            if run_id >= args_cli.log_tilt and args_cli.log_tilt > 0:
                print("Finished requested tilt logging runs.")
                break

            if logged_samples == args_cli.log_io and not done_sampling:
                print("Finished io sampling.")
                io_log_inputs = os.path.join(io_log_folder, "nn_inputs.npy")
                io_log_outputs = os.path.join(io_log_folder, "nn_outputs.npy")
                print("Inputs: ", input_log[:5])
                print("Outputs: ", output_log[:5])
                np.save(io_log_inputs, input_log)
                np.save(io_log_outputs, output_log)
                done_sampling = True

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
