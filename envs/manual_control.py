# Taken from https://github.com/Farama-Foundation/gym-minigrid/blob/master/manual_control.py
import time
import hydra
import numpy as np
import gym
from gym_minigrid.window import Window
import synthetic_envs

import datetime
from PIL import Image


@hydra.main(config_path="../configs/env", config_name="push-N2C3S1S1")
def main(config):
    env = getattr(synthetic_envs, config.env)(config, seed=0)

    def redraw(img):
        window.show_img(env.render())

    def reset():
        obs = env.reset()
        redraw(obs)

    def step(action):
        obs, reward, done, info = env.step(action)
        print("step=%s, reward=%.2f" % (env.step_count, reward))
        im = Image.fromarray(obs)
        im_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        im.save(f"{im_name}.png")
        if done:
            print("done!")
            reset()
        else:
            redraw(obs)

    def key_handler(event):
        print("pressed", event.key)
        if event.key == "escape":
            window.close()
            return
        if event.key == "backspace":
            reset()
            return
        if event.key == "up":
            step(0)
            return
        if event.key == "left":
            step(1)
            return
        if event.key == "down":
            step(2)
            return
        if event.key == "right":
            step(3)
            return

    window = Window(config.name)
    window.reg_key_handler(key_handler)

    reset()

    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    main()
