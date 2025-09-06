## 1. Overview

This is the Pytorch implementation for "Robust Model for Speech Command Recognition with Integrated Classifier (xxx24)"

Authors: Van-Hoa Nguyen, Viet-Cuong Ta

Paper: [Robust Model for Speech Command Recognition with Integrated Classifier](https://www.xxxxx)

![Proposed Architecture](img/proposed_architecture.pdf)

## 2. Setup

### 2.1 Environment
`pip install -r requirements.txt`

### 2.2 Datasets
The data will be automatically downloaded and processed

## 3. Experiments

3.0 To run IntergratedNetwork:

```python main.py```

3.1 Seed: 42

3.2 Dataset choices: [google/speech_commands](https://huggingface.co/datasets/google/speech_commands)

## 4. Backbone
- [wav2vec2-base](https://github.com/TaiLvYuanLiang/HGWaveNet)
- [wav2vec2-base-960h](https://github.com/marlin-codes/HTGN)
- [wav2vec2-large](https://github.com/VGraphRNN/VGRNN)
- [wavlm-base](https://github.com/IBM/EvolveGCN)
- [wavlm-large](https://github.com/FeiGSSS/DySAT_pytorch)