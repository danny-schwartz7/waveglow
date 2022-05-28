import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read
import librosa

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

from mel2samp import Mel2Samp


def filter_audio(hf_item, min_samples, target_sampling_rate):
    # This fixes https://github.com/NVIDIA/waveglow/issues/95 which created instability during training
    
    # original_sampling_rate = hf_item['audio']['sampling_rate']min_samples / target_sampling_rate * original_sampling_rate
    if hf_item['audio']['array'].shape[-1] < 22050:
        return False
    return True

class LibriDataset(Mel2Samp):
    def __init__(self, hf_ds, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        hf_ds = hf_ds.filter(filter_audio, fn_kwargs={
            'min_samples': segment_length,
            'target_sampling_rate': sampling_rate
        })
        self.hf_ds = hf_ds.shuffle(seed=1234)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        hf_item = self.hf_ds[index]
        audio = hf_item['audio']['array']
        sampling_rate = hf_item['audio']['sampling_rate']
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        audio = torch.from_numpy(audio).float()

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            raise NotImplementedError("Audio too short")
            # audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return len(self.hf_ds)