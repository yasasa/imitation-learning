import h5py
import numpy as np
from agents.imitation.imitation_learning import ImitationLearning
learner = ImitationLearning("Town01", False)

data = h5py.File("/localdata2/yasasaa/dataset1.h5")
images = list(data["images"])
outputs = list(data["outputs"])
speeds = list(data["speeds"])
cmds = [2.] * len(images)

learner.train_model(images, speeds, cmds, outputs, epochs=500)



