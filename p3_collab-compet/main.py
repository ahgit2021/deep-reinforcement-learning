from unityagents import UnityEnvironment
import numpy as np
import train
import importlib
import sys
importlib.reload(train)

env = UnityEnvironment(file_name="Tennis.app")

train.train_agent(env, int(sys.argv[1]))

env.close()
