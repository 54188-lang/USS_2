from pathlib import Path
from typing import Dict, Sequence, Union
from dataclasses import dataclass, field, asdict

@dataclass
class Configurations(object):
    # Model hyperparameters
    model_mic_num: int = 2
    model_ch_dim: int = 8
    model_enc_dim: int = 512
    model_feature_dim: int = 128
    model_win: int = 16
    model_layer: int = 8
    model_stack: int = 1
    model_kernel: int = 3
    model_num_spk: int = 4
    model_causal: bool = False

config = Configurations()
