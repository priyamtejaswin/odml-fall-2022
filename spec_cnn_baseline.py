import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

commands = ['one', 'down', 'dog', 'three', 'up', 'yes', 'stop', 'seven', 'left',
 'wow', 'eight', 'two', 'right', '_silence_', 'no', 'happy', 'nine', 'zero', 'cat', 
 'off', 'sheila', 'marvin', 'tree', 'four', 'bird', 'bed', 'on', 'go', 'five', 
 'six', 'house']
cm2ix = {n:i for i, n in enumerate(commands)}

def load_audio_file(path, sr=16000, length=1.0):
    """
    sr: Desired sampling rate.
    Only used if sr and path-sr don't match.
    
    length: Desired audio length.
    If more than recorded audio, waveform is zero-padded.
    """
    samples = int(sr * length)
    waveform, original_rate = torchaudio.load(path, normalize=True)
    
    if sr != original_rate:
        waveform = torchaudio.functional.resample(waveform, original_rate, sr)
        
    size = waveform.shape[1]
    if size == samples:
        return waveform
    
    if samples < size:
        return waveform[:, :samples]
    
    return torch.nn.functional.pad(waveform, (0, samples-size))

def collect_pairs(path):
    assert os.path.isdir(path), "Path must be a dir!"
    x = []
    y = []
    for l in commands:
        assert l in os.listdir(path), "Label type %s not found in path." % l
        c = 0
        for f in os.listdir(os.path.join(path, l)):
            if f.endswith('.wav'):
                p = os.path.join(path, l, f)
                x.append(p)
                y.append(cm2ix[l])
                
                c += 1
                
        print("Found %d files for label %s." % (c, l))
        
    print("Found %d files." % len(x))
    return x, y

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels):
        assert len(files) == len(labels)
        self.files = files
        self.labels = labels
        self.fspec = torchaudio.transforms.Spectrogram(n_fft=256, power=2)
        
    def __getitem__(self, ix):
        path = self.files[ix]
        label = self.labels[ix]
        # return fspec(load_audio_file(path)), label
        return {"spec": torch.absolute(self.fspec(load_audio_file(path))), 
                "label": label}
    
    def __len__(self):
        return len(self.files)
    
def create_model(pretrained=None):
    model = torch.nn.Sequential(
        torch.nn.Upsample(size=(64, 64)),
        # torch.nn.BatchNorm2d(1),  # Single channel.
        torch.nn.Conv2d(1, 64, 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d((2, 2)),
        
        torch.nn.Conv2d(64, 64, 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d((2, 2)),
        
        torch.nn.Conv2d(64, 128, 3),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d((2, 2)),
        
        torch.nn.Conv2d(128, 128, 3),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d((2, 2)),
        
        # torch.nn.Dropout(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(inplace=True),
        # torch.nn.Dropout(0.5),
        torch.nn.Linear(128, len(cm2ix))
    )
    
    if pretrained is None:
        return model
    else:
        print("Trying to load prameters.")
        assert os.path.isfile(pretrained), "Pre-trained file does not exist."
        model.load_state_dict(torch.load(pretrained, map_location=device))
        print("Loaded successfully from", pretrained)
        return model
    
def train(epoch, model, optimizer, lfn, dataiter, device):
    print("Epoch", epoch)
    epoch_loss = 0.0
    
    for batch in tqdm(dataiter):
        x, y = batch["spec"], batch["label"]
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        output = model(x)
        loss = lfn(output, y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    epoch_loss /= len(dataiter)
    print("Loss", np.round(epoch_loss, 3))
    return epoch_loss

def evalacc(model, dataiter, device):
    correct, size = 0, 0
    for batch in tqdm(dataiter):
        x, y = batch["spec"], batch["label"]
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            model.eval()
            outputs = model(x).cpu().numpy()
            preds = np.argmax(outputs, -1)
            correct += np.sum(preds == y.cpu().numpy())
            size += len(preds)
            
    accuracy = correct/size
    return accuracy, size