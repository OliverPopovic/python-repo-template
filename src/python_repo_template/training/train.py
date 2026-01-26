from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    #Print the fully-resolved config Hydra loaded
    print(OmegaConf.to_yaml(cfg))

    #Show where Hydra is running
    print("cwd:", os.getcwd())

    #Example
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("output_dir: ", out_dir.resolve())


if __name__ == "__main__":
    main()