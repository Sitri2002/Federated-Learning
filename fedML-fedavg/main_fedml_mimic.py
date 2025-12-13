import os
import sys
import yaml
import torch

import fedml
from fedml import FedMLRunner

from data_mimic import load_partition_data_mimic
from model_resnet_mimic import create_model
from trainer_mimic import MIMICTrainer
from fedml.ml.trainer import trainer_creator
from model_trainer_mimic import ModelTrainerMIMIC



def _deep_get(d, paths, default=None):
    for p in paths:
        cur = d
        ok = True
        for k in p.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_args(args, cfg):
    def set_if_missing(k, v):
        if not hasattr(args, k):
            setattr(args, k, v)


    set_if_missing("training_type", _deep_get(cfg, ["common_args.training_type"], "simulation"))
    set_if_missing("backend", _deep_get(cfg, ["comm_args.backend"], "sp"))

    set_if_missing("dataset", _deep_get(cfg, ["data_args.dataset"], "mimic"))

    set_if_missing("federated_optimizer", _deep_get(cfg, ["train_args.federated_optimizer"], "FedAvg"))
    set_if_missing("client_optimizer", _deep_get(cfg, ["train_args.client_optimizer"], "adam"))
    set_if_missing("learning_rate", float(_deep_get(cfg, ["train_args.learning_rate"], 1e-4)))
    set_if_missing("lr", float(_deep_get(cfg, ["train_args.lr"], getattr(args, "learning_rate", 1e-4))))

    set_if_missing("epochs", int(_deep_get(cfg, ["train_args.epochs"], 1)))
    set_if_missing("batch_size", int(_deep_get(cfg, ["train_args.batch_size"], 8)))
    set_if_missing("weight_decay", float(_deep_get(cfg, ["train_args.weight_decay"], 0.0)))
    set_if_missing("momentum", float(_deep_get(cfg, ["train_args.momentum"], 0.9)))

    set_if_missing("frequency_of_the_test", int(_deep_get(cfg, ["validation_args.frequency_of_the_test"], 1)))

    # data paths
    set_if_missing("data_cache_dir", _deep_get(cfg, ["data_args.data_cache_dir"], "data_cache"))
    set_if_missing("manifest_path", _deep_get(cfg, ["data_args.manifest_path"], None))
    set_if_missing("image_dir", _deep_get(cfg, ["data_args.image_dir"], None))

    return args



def ensure_cfg(cfg_path):
    if not any(a.startswith("--cf") or a.startswith("--yaml_config_path") for a in sys.argv):
        sys.argv += ["--cf", cfg_path]


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base, "config", "fedml_config.yaml")

    fedml._global_training_type = "simulation"
    fedml._global_comm_backend = "sp"

    ensure_cfg(cfg_path)

    args = fedml.init()
    cfg = _load_yaml(cfg_path)
    args = normalize_args(args, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() and args.using_gpu else "cpu")

    dataset = load_partition_data_mimic(args)

    model = create_model(args).to(device)
    trainer = MIMICTrainer(model, args)

    _orig_create = trainer_creator.create_model_trainer

    def _patched_create_model_trainer(model, args):
        if getattr(args, "dataset", None) == "mimic":
            return ModelTrainerMIMIC(model, args)
        return _orig_create(model, args)

    trainer_creator.create_model_trainer = _patched_create_model_trainer

    runner = FedMLRunner(args, device, dataset, model, trainer)
    runner.run()