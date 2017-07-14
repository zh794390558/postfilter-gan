#!/usr/bin/env python

import librosa
import os
import numpy as np

# mel-scpetrum
def windows(signal, window_size, stride=0.5):
        """ Return windows of the given signal 
            by sweeping in stride fractions of window
        """
        assert signal.ndim == 1, signal.ndim
        n_samaples = signal.shape[0]
        offset = int(window_size * stride)
        for beg_i, end_i in zip(range(0, n_samples, offset),
                                range(window_size, n_samples + offset, offset)):
                if end_i - beg_i < window_size:
                        break
                slice_ = signal[beg_i:end_i]
                if slice_.shape[0] == window_size:
                        yield slice_


def extract_features(wav_file, bands=40, frames=40, frame_length=512, rate=22050):
        window_size = frame_length * (frames -1)
        log_specgrams = []
        y, sr = librosa.load(f)
        if sr != rate:
                raise ValueError('Sampling rate is expected to be 22.5kH!')

        for signal in windows(y, window_size):
                melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=bands)
                logsepc = librosa.logamplitude(melspec) # [n_mel, frames]
                logsepc = logspec[:, np.newaxis] # [n_mel, frames, channels]
                log_specgrams.append(logspec)

        log_specgrams = np.asarray(log_specgrams) # [batch, n_mel ,frames, channels]
        return log_specgrams

