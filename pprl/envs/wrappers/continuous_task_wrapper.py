import gymnasium as gym


class ContinuousTaskWrapper(gym.Wrapper):
    """This wrapper forces terminated to be always False."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, False, truncated, info
