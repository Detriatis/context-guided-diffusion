from dataclasses import dataclass, field, asdict, is_dataclass, fields
from typing import Callable, Any, Dict, Literal, Type, TypeVar
from swiss_roll.utils import sample_uniform
from pathlib import Path 

import torch
import yaml 

@dataclass
class OptParams:
    lr: float = 1e-2 
    weight_decay: float = 0.0
    l2_lambda: float = 1e3

@dataclass
class CGDParams:
    target_mean: float = -0.368
    target_logvar: float = 0.7
    sigma_t: float = 1.0 
    tau_t: float = 0.1
    reg_lambda: float = 1.0
    ctx_size: int = 512
    reg_scale_by: Literal["ctx_size", "dataset_size"] = "ctx_size"

@dataclass 
class TrainParams:
    batch_size: int = 128
    n_epochs: int = 30 

@dataclass
class RunParams:
    expose_time: bool = True
    reduction: str = 'mean'
    run_id: int = 0000
    reg_type: Literal["mse", "cgd"] = "cgd"
    update_embeds: bool = False

@dataclass
class SamplerParams:
    sampler_func: Callable[..., torch.Tensor] = sample_uniform
    sampler_args: Dict[str, Any] = field(default_factory=lambda: {"low": -2.5, "high": 2.5})
    sample_shape: tuple[int, int] = (10_000, 2)


@dataclass 
class Config:
    opt: OptParams
    cgd: CGDParams
    train: TrainParams
    run: RunParams
    samp: SamplerParams

@dataclass
class HyperOptConfig:
    cfg: Config
    startup: int = 10,
    seed: int = 42, 
    trials: int = 60,
    timeout: int = 12 * 60 * 60
    pareto: bool = False

@dataclass 
class Metrics:
    nll: float
    mse: float 
    loss: float 
    l2: float
    cov: float

@dataclass
class TrainValMetrics:
    train: Metrics
    val: Metrics 

T = TypeVar("T")
def from_dict(dataclass_type: Type[T], raw: dict) -> T:
    if not is_dataclass(dataclass_type): 
        return raw
    
    kwargs = {}
    for f in fields(dataclass_type):
        name, typ = f.name, f.type 
        if name not in raw:
            continue 
        value = raw[name]
        if is_dataclass(typ) and isinstance(value, dict): 
            kwargs[name] = from_dict(typ, value)
        else: 
            kwargs[name] = value
    return dataclass_type(**kwargs) 

def load_config(path) -> Config:
    with open(path, 'r') as f: 
        raw_cfg = yaml.load(f, yaml.FullLoader)
    
    return from_dict(raw_cfg) 

def load_hyperoptconfig(path: Path) -> HyperOptConfig:
    
    with open(path, 'r') as f: 
        raw = yaml.load(f, yaml.FullLoader)
    raw['cfg'] = from_dict(Config, raw['cfg'])
    
    return from_dict(HyperOptConfig, raw) 

def save_config(cfg, path: Path): 
    cfg_dict = asdict(cfg) 
    with open(path.with_suffix('.yaml'), 'w') as f:
        yaml.dump(cfg_dict, f)