import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class AudioCapsDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, sample_rate=16000, duration=10.0):
        self.metadata = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = os.path.join(self.audio_dir, row['audio_filename'])
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate to target length
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)  # Log mel spectrogram
        
        # Get text caption
        caption = row['caption']
        
        return mel_spec, caption, self.target_length