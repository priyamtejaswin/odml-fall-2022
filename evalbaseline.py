import os
import sys
import numpy as np
import torch
import torchaudio
from spec_cnn_baseline import commands
from spec_cnn_baseline import cm2ix
from spec_cnn_baseline import AudioDataset
from spec_cnn_baseline import collect_pairs
from spec_cnn_baseline import create_model
from spec_cnn_baseline import evalacc

# Set the seed value for experiment reproducibility.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

print("Valid commands:")
print(cm2ix)

# Checks
testdir = sys.argv[1]
print("Provided path", testdir)
assert os.path.isdir(testdir), "Path is not a valid dir."
for d in os.listdir(testdir):
    assert os.path.isdir(os.path.join(testdir, d)), "Path should only consider dirs. Found file %s" % d
    assert d in cm2ix, "Found label that is not part of trained labels %s" % d

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

test_dset = AudioDataset(*collect_pairs(testdir))
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=64, shuffle=False)

_batch = next(iter(test_loader))
print("Spec:", _batch["spec"].shape)
print("Label:", _batch["label"].shape)

# Model
model = create_model("sc_v01_trained (2).pt")

# Eval
print("Testing (accuracy, samples)")
print(evalacc(model, test_loader, device))

# Parameters
print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
