"""LightX2V training entrypoint with H3 A100 extensions registered."""

from __future__ import annotations

import argparse

import torch
from loguru import logger

# Importing these modules registers the custom model and trainer names before
# LightX2V resolves the YAML configuration.
from . import model as _model  # noqa: F401
from . import trainer as _trainer  # noqa: F401
from .distributed import init_distributed_a100

from lightx2v_train.data import build_data, prepare_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, load_config, setup_logger
from lightx2v_train.trainers import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniMax-H3 DMD on A100 clusters")
    parser.add_argument("--config", required=True, help="Path to the A100 YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    init_distributed_a100(config)
    setup_logger(config)

    try:
        prepare_data(config)
        model = build_model(config)
        model.load_components()

        dataloader_train = build_data(config, train_or_val="train")
        dataloader_eval = None
        if config.get("inference", {}).get("infer_every_iters", None):
            dataloader_eval = build_data(config, train_or_val="val")

        trainer = build_trainer(config)
        trainer.set_model(model)
        trainer.set_data(dataloader_train, dataloader_eval)
        trainer.train()
    except Exception:
        logger.exception("H3 A100 training failed")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
