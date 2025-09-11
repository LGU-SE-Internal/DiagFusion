#!/usr/bin/env -S uv run -s
from rcabench_platform.v2.online.run_exp_platform import run
from rcabench_platform.v2.algorithms.spec import global_algorithm_registry
from src.diagfusion.DiagfusionClass import diagfusion

if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["diagfusion"] = diagfusion

    run(enable_builtin_algorithms=False)
