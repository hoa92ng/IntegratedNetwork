import librosa
import numpy as np
import torchaudio.transforms as T
from scipy.signal import butter, lfilter
from enum import Enum
import random 
import soundfile as sf
import acoustics
from datasets import Dataset, Audio, ClassLabel, concatenate_datasets, Value

class AugmentationMethod(Enum):
    NONE = 0
    TIME_STRETCH = 1
    PITCH_SHIFT = 2
    ADD_NOISE = 3
    CHANGE_VOLUME = 4
    REVERB = 5
    SHIFT_TIME = 6

def time_stretch(audio, rate=1.2):
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.05):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def shift_time(audio, shift_max=0.2, sr=16000):
    shift = np.random.randint(sr * shift_max) 
    return np.roll(audio, shift)

def change_volume(audio, factor=1.5):    
    return audio * factor

def spec_augment(mel_spectrogram):
    time_mask = T.TimeMasking(time_mask_param=30)
    freq_mask = T.FrequencyMasking(freq_mask_param=30)
    augmented_spec = time_mask(mel_spectrogram)
    augmented_spec = freq_mask(augmented_spec)
    return augmented_spec

def reverb(audio, sr=16000):
    room_size = 0.5
    return librosa.effects.preemphasis(audio, coef=room_size)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(audio, cutoff, sr):
    b, a = butter_lowpass(cutoff, sr, order=6)
    y = lfilter(b, a, audio)
    return y

def get_silence_data_resources(NOISE_AUDIO_PATH_ARRAYS:list):
    data_bank = []
    for path in NOISE_AUDIO_PATH_ARRAYS:
        audio_data, sample_rate = sf.read(path)
        data_bank.append((audio_data, sample_rate))
    return data_bank

def random_crop_wav(data_bank, idx, crop_duration=1):
    audio_data, sample_rate = data_bank[idx]
    total_duration = len(audio_data) / sample_rate
    
    if crop_duration > total_duration:
        raise ValueError("Crop duration exceeds the audio length.")

    crop_samples = int(crop_duration * sample_rate)
    max_start = len(audio_data) - crop_samples
    start_sample = random.randint(0, max_start)
    cropped_audio = audio_data[start_sample:start_sample + crop_samples]
    cropped_array = np.array(cropped_audio)
    return cropped_array, sample_rate

def create_synthetic_silence_data(NOISE_AUDIO_PATH_ARRAYS:list=None, NOISE_AUDIO_ARRAYS:list=None, data_count=1000):
    """
    Generates a list of dictionaries containing noise data.
    Each dictionary is ready to be used to create a Hugging Face Dataset.
    """
    output_data_list = []
    sample_rate = 16_000
    if NOISE_AUDIO_PATH_ARRAYS:
        data_bank = get_silence_data_resources(NOISE_AUDIO_PATH_ARRAYS)
    else:
        data_bank = [(x, sample_rate) for x in NOISE_AUDIO_ARRAYS]
    
    for i in range(data_count):
        rand_method = random.randint(0, 1)
        # Crop from data bank
        if rand_method == 0:
            arr, sr = random_crop_wav(data_bank=data_bank, idx=random.randint(0, 3))
            arr = arr
        # Generate new noise
        else:
            rand_noise_method = random.randint(0, 1)     
            # White noise
            if rand_noise_method == 1:
                # Generate white noise array directly
                noise_array = acoustics.generator.noise(sample_rate * 1, color='white')
            # Pink noise
            else:
                # Generate pink noise array directly
                noise_array = acoustics.generator.noise(sample_rate * 1, color='pink')
            arr = np.array((noise_array / 3) * 32767).astype(np.int16)
            sr = sample_rate

        data_dict = {
            'file': f'added_synthetic_audio_file_{i}',
            'audio': {
                'path': f'added_synthetic_audio_file_{i}',
                'array': arr,
                'sampling_rate': sr
            },
            # Using 10 as the label for noise/silence
            'label': 10,
            "is_unknown": False,
            "speaker_id": "Synthetic",
            "utterance_id": -1
        }
        output_data_list.append(data_dict)
    return output_data_list

def create_and_combine_datasets(concating_dataset, noise_data_count=1800, NOISE_AUDIO_PATH_ARRAYS:list=None, NOISE_AUDIO_ARRAYS:list=None):
    """
    Combines the process of creating noise data and merging it
    with an existing dataset.
    """
    synthetic_silence_data = create_synthetic_silence_data(NOISE_AUDIO_PATH_ARRAYS, NOISE_AUDIO_ARRAYS, data_count=noise_data_count)

    new_synthetic_silence_dataset = Dataset.from_list(synthetic_silence_data)

    new_synthetic_silence_dataset = new_synthetic_silence_dataset.cast_column('audio', Audio(sampling_rate=16_000))
    new_synthetic_silence_dataset = new_synthetic_silence_dataset.cast_column(
        'label', 
        ClassLabel(names=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_silence_', 'unknown'])
    ).cast_column("label", Value("int64"))
    new_synthetic_silence_dataset = new_synthetic_silence_dataset.cast_column("utterance_id", Value("int8"))  # <- fix the error


    final_dataset = concatenate_datasets([concating_dataset, new_synthetic_silence_dataset])
    return final_dataset

def random_augementation(audio, rand_method:AugmentationMethod=AugmentationMethod.NONE):
    if rand_method is AugmentationMethod.NONE:   
        rand_method = AugmentationMethod(random.randint(1, 5))
    if rand_method == AugmentationMethod.TIME_STRETCH:
        return time_stretch(audio=audio)
    elif rand_method == AugmentationMethod.PITCH_SHIFT:
        return pitch_shift(audio=audio, sr=16_000)
    elif rand_method == AugmentationMethod.ADD_NOISE:
        noise_fc = random.uniform(0.001, 0.005)
        return add_noise(audio=audio, noise_factor=noise_fc)
    elif rand_method == AugmentationMethod.CHANGE_VOLUME:
        return change_volume(audio=audio)
    elif rand_method == AugmentationMethod.REVERB:
        return reverb(audio=audio, sr=16_000)
    elif rand_method == AugmentationMethod.SHIFT_TIME:
        return shift_time(audio=audio, sr=16_000)
