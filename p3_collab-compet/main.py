from unityagents import UnityEnvironment
import numpy as np
import train
import importlib
importlib.reload(train)

env = UnityEnvironment(file_name="Tennis.app")

train.train_agent(env, 30)

env.close()
