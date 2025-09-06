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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--ws-url", type=str, default="ws://localhost:8765", help="WebSocket server URL.")
parser.add_argument("--log-tilt", type=int, default=0, help="Number of timesteps to log robot tilt (0 = disable).")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import asyncio
import websockets
import json
import os
import numpy as np

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import quat_apply

# ----------------- WebSocket -----------------
class WebSocketClient:
    def __init__(self, url):
        self.url = url
        self.websocket = None
    
    async def connect(self):
        self.websocket = await websockets.connect(self.url)
        print(f"Connected to WebSocket server at {self.url}")
    
    async def send_observations(self, obs):
        obs_str = ",".join(str(x.item()) for x in obs.flatten())
        await self.websocket.send(obs_str)
    
    async def receive_actions(self):
        action_str = await self.websocket.recv()
        try:
            return torch.tensor(json.loads(action_str))
        except json.JSONDecodeError:
            return torch.tensor(eval(action_str))
    
    async def close(self):
        if self.websocket:
            await self.websocket.close()

# ----------------- Main -----------------
@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]

    if args_cli.log_tilt > 0:
        # folder relative to the script file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "../../../tilt_logging_quant")
        os.makedirs(folder, exist_ok=True)

    ws_client = WebSocketClient(args_cli.ws_url)

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


    async def simulation_loop():
        await ws_client.connect()

        run_id = 0
        obs, _ = env.reset()
        timestep = 0

        time_log = []
        tilt_log = []

        try:
            while simulation_app.is_running():
                start_time = time.time()

                with torch.inference_mode():
                    await ws_client.send_observations(obs["policy"][0])
                    actions = await ws_client.receive_actions()
                    if actions.dim() == 1:
                        actions = actions.unsqueeze(0)
                    obs, _, _, _, _ = env.step(actions)

                # log tilt if within run count
                if run_id < args_cli.log_tilt:
                    tilt_deg = compute_tilt()[0].item() * 180.0 / 3.14159
                    sim_time = timestep * dt
                    time_log.append(sim_time)
                    tilt_log.append(tilt_deg)

                # sleep for real-time sim
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

                timestep += 1

                if timestep >= env.unwrapped.max_episode_length:
                    run_id += 1
                    if run_id <= args_cli.log_tilt:
                        filename = os.path.join(folder, f"tilt_log_run{run_id}.npz")
                        np.savez(filename, time=np.array(time_log), tilt=np.array(tilt_log))
                        print(f"Saved tilt log to {filename}")
                    # reset logs for next loop
                    time_log, tilt_log = [], []
                    timestep = 0

                    # stop logging if finished required runs
                    if run_id >= args_cli.log_tilt and args_cli.log_tilt > 0:
                        print("Finished requested tilt logging runs.")
                        break

        finally:
            await ws_client.close()
            env.close()

    asyncio.run(simulation_loop())

if __name__ == "__main__":
    main()
    simulation_app.close()
