### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import imageio
import os
import numpy as np
import sys
import wandb
import time
import os

class VideoRecorder(object):
    def __init__(self, save_path, height=256, width=256, camera_id=0, fps=30):
        # self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.save_dir = os.path.join(save_path, "videos")
        os.makedirs(self.save_dir)
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array',height=self.height, width=self.width, camera_id=self.camera_id)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            self.save_to_wand_as_gif()

    def save_to_wand_as_gif(self):
        FPS = 4
        video_array = np.stack(self.frames, axis=0)
        video_array = np.moveaxis(video_array, 3, 1)
        wandb.log({"video": wandb.Video(video_array, fps=FPS, format="gif")})
