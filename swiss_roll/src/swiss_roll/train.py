import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json 
from pathlib import Path
from torch.utils.data import DataLoader 
from swiss_roll import DATA_DIR
from swiss_roll.diffusion import DiffusionModel
from swiss_roll.datagen import load_swissroll
from swiss_roll.guidance import cgd_regularization_term, sample_uniform, Data, GuidanceModel, ContextEmbedding