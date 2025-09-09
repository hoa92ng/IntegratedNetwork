from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset, concatenate_datasets
from src.data.audio_augmentation import create_synthetic_datasets, random_augementation, AugmentationMethod

# Keywords
LABEL_MAP = {
    'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4,
    'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9,
    'unknown': 10, '_silence_': 11
}

ANOMALY_LABELS = {10: 0, 11: 10}
VALID_LABELS = list(range(10))
VALID_LABELS = list(range(10))

def transform_labels(examples: dict) -> dict:
    """
    Transforms labels in a batch of examples.
    - Maps 'unknown' and '_silence_' to their correct integer values.
    """
    labels = examples['label']
    is_unknown = examples['is_unknown']
    
    transformed_labels = []
    for label, unknown in zip(labels, is_unknown):
        # Convert integer label back to its name using the dataset's features
        label_name = ID_TO_LABEL.get(label)
        
        if unknown:
            transformed_labels.append(LABEL_MAP['unknown'])
        elif label_name == '_silence_':
            transformed_labels.append(LABEL_MAP['_silence_'])
        else:
            transformed_labels.append(label)
            
    examples['label'] = transformed_labels
    return examples

def add_nomaly_and_re_labels(examples: dict) -> dict:
    """
    Adds 'nomaly_label' and 're_label' based on the 'label' field.
    'nomaly_label' is 0 for 'unknown' and 1 otherwise.
    're_label' is -1 for 'unknown', 10 for '_silence_', and the original label otherwise.
    """
    labels = examples['label']
    
    nomaly_labels = []
    re_labels = []

    for label in labels:
        if label == LABEL_MAP['unknown']:
            nomaly_labels.append(0)
            re_labels.append(-1)
        else:
            nomaly_labels.append(1)
            re_labels.append(ANOMALY_LABELS.get(label, label))

    examples['nomaly_label'] = nomaly_labels
    examples['re_label'] = re_labels
    return examples


class CustomAudioDataset(Dataset):
    """
    Specific for SpeechCommand Dataset.
    The modified dataset include a 'normal' property.
    Thin PyTorch Dataset wrapper around a HuggingFace Dataset on disk.
    - Applies a HF feature_extractor to raw audio and stores padded 'input_values'
    - Adds 'anomaly_label' (1 = normal, 0 = unknown) for two-head training
    - Optional waveform transforms (e.g., augmentation) can be injected
    """
    def __init__(
        self,
        dataset_path: str,
        feature_extractor,
        use_transform: bool = False,
        use_data_from_disk: bool = True,
        data_version: str = 'v0.01',
        split: str = 'train',
        sr: int = 16_000,
        use_augmentation: bool = True,
        transforms: Optional[Sequence[Callable[[np.ndarray, int], np.ndarray]]] = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.sample_rate = sr
        self.use_transform = use_transform
        self.transforms = transforms or []

        if use_data_from_disk:
            ds = load_from_disk(dataset_path)
        else:
            ds = load_dataset("google/speech_commands", data_version, split=split, trust_remote_code=True)
        
         # Global access to the label mapping, useful for transform_labels
        global ID_TO_LABEL, LABEL_TO_ID
        class_names = ds.features['label'].names
        LABEL_TO_ID = {label: i for i, label in enumerate(class_names)}
        ID_TO_LABEL = {i: label for i, label in enumerate(class_names)}
        ds = ds.map(transform_labels, batched=True).map(add_nomaly_and_re_labels, batched=True)
        ds = ds.remove_columns('label').rename_column('re_label', 'label')

        if use_augmentation:
            # Augementation dataset
            # Silent generation 5 -> 1800
            # Filter rows with label == 10
            # filtered_silence_dataset = ds.filter(lambda example: example["label"] == LABEL_TO_ID['_silence_'])
            filtered_silence_dataset = ds.filter(lambda example: example["label"] == 10)
            # Convert audio to list of numpy arrays
            audio_list = [np.array(sample["audio"]["array"]) for sample in filtered_silence_dataset]
            
            synthetic_dataset = create_synthetic_datasets(NOISE_AUDIO_ARRAYS=audio_list)
            synthetic_dataset = synthetic_dataset.map(transform_labels, batched=True).map(add_nomaly_and_re_labels, batched=True)
            synthetic_dataset = synthetic_dataset.remove_columns('label').rename_column('re_label', 'label')
            ds = concatenate_datasets([ds, synthetic_dataset])

        ds = ds.map(self._preprocess_function, remove_columns=["audio"], batched=True)
        ds = ds.map(self._add_anomaly_label)
        self.ds = ds

    def _preprocess_function(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_arrays = [x["array"] for x in batch["audio"]]
        out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            padding="max_length",
            max_length=16_000,
            truncation=True,
        )
        return out  # contains 'input_values' (List[List[float]])

    def _add_anomaly_label(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # In speechcommand dataset, anomaly (unknown) class is 11
        item["anomaly_label"] = 0 if item["label"] == 11 else 1
        return item

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ds[idx]
        ivals = np.asarray(row["input_values"], dtype=np.float32)

        if self.use_transform and self.transforms:
            # choose a transform or leave unchanged
            t_idx = random.randint(0, len(self.transforms))
            if t_idx < len(self.transforms):
                ivals = self.transforms[t_idx](ivals, self.sample_rate)

        return {
            "label": torch.as_tensor(row["label"], dtype=torch.long),
            "nomaly_label": torch.as_tensor(row["nomaly_label"], dtype=torch.long),
            "input_values": torch.from_numpy(ivals),
        }

    # Utility for quick counts (optional)
    def show_counts(self) -> None:
        df = self.ds.to_pandas()
        print(df["label"].value_counts())




