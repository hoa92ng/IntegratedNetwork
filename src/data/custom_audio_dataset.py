from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

# Keywords
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
UNKNOWN_WORDS_V1 = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]
UNKNOWN_WORDS_V2 = UNKNOWN_WORDS_V1 + ["backward", "forward", "follow", "learn", "visual"]
SILENCE = "_silence_"
LABELS_V1 = WORDS + UNKNOWN_WORDS_V1 + [SILENCE]
LABELS_V2 = WORDS + UNKNOWN_WORDS_V2 + [SILENCE]


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
        sr: int = 16_000,
        transforms: Optional[Sequence[Callable[[np.ndarray, int], np.ndarray]]] = None,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.sample_rate = sr
        self.use_transform = use_transform
        self.transforms = transforms or []

        ds = load_from_disk(dataset_path)
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
            "anomaly_label": torch.as_tensor(row["anomaly_label"], dtype=torch.long),
            "input_values": torch.from_numpy(ivals),
        }

    # Utility for quick counts (optional)
    def show_counts(self) -> None:
        df = self.ds.to_pandas()
        print(df["label"].value_counts())