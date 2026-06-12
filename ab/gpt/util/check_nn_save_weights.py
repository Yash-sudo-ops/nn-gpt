"""
Train via nn-dataset with save_pth_weights=True and copy best_model.pth into save_path/weights.pth.

NNEval's default api.check_nn passes save_pth_weights=False, so no local checkpoint is written.
Use this when --mobile_deploy / --save_weights_pth is enabled.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

import ab.nn.util.CodeEval as codeEvaluator
from ab.nn.util.Const import ab_root_path, ckpt_dir, out
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Train import Train
from ab.nn.util.Util import create_file, remove, release_memory, uuid4


def _checkpoint_path(model_name: str) -> Path:
    short = model_name.split(".")[-1]
    return Path(ckpt_dir) / short / "best_model.pth"


def _copy_to_weights_pth(model_name: str, save_path: Union[str, Path]) -> Optional[Path]:
    src = _checkpoint_path(model_name)
    if not src.exists():
        return None
    dest_dir = Path(save_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "weights.pth"
    state = torch.load(src, map_location="cpu")
    torch.save(
        {
            "state_dict": state if not isinstance(state, dict) or "state_dict" not in state else state["state_dict"],
            "weights_source": str(src),
            "model_name": model_name,
            "trained": True,
        },
        dest,
    )
    return dest


def check_nn_save_pth(
    nn_code: str,
    task: str,
    dataset: str,
    metric: str,
    prm: dict,
    save_to_db: bool = True,
    prefix: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    export_onnx: bool = False,
    epoch_limit_minutes=None,
    transform_dir=None,
) -> Tuple[str, float, float, float]:
    """Same contract as ab.nn.api.check_nn but saves best_model.pth and copies to save_path/weights.pth."""
    from ab.nn.util.Const import default_epoch_limit_minutes
    from ab.nn.util.db import Write as DB_Write
    from ab.nn.util.Util import good

    if epoch_limit_minutes is None:
        epoch_limit_minutes = default_epoch_limit_minutes

    model_name = uuid4(nn_code)
    if prefix:
        model_name = f"{prefix}-{model_name}"

    tmp_modul = ".".join((out, "nn", "tmp"))
    tmp_modul_name = ".".join((tmp_modul, model_name))
    tmp_dir = ab_root_path / tmp_modul.replace(".", "/")
    create_file(tmp_dir, "__init__.py")
    temp_file_path = tmp_dir / f"{model_name}.py"
    trainer = None
    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(nn_code)
        res = codeEvaluator.evaluate_single_file(temp_file_path)
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(
            task, dataset, prm["transform"], transform_dir
        )
        num_workers = prm.get("num_workers", 1)
        trainer = Train(
            config=(task, dataset, metric, model_name),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm["batch"],
            nn_module=tmp_modul_name,
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=num_workers,
            prm=prm,
            save_to_db=save_to_db,
            is_code=True,
        )
        epoch = prm["epoch"]
        accuracy, accuracy_to_time, duration = trainer.train_n_eval(
            epoch,
            epoch_limit_minutes,
            True,  # save_pth_weights
            export_onnx,
            train_set,
            save_path=save_path,
        )
        if save_path:
            copied = _copy_to_weights_pth(model_name, save_path)
            if copied:
                print(f"[weights] Saved trained checkpoint -> {copied}")
            else:
                print(f"[weights] WARN: best_model.pth not found under {ckpt_dir}")
        if save_to_db:
            if good(accuracy, minimum_accuracy, duration):
                model_name = DB_Write.save_nn(
                    nn_code, task, dataset, metric, epoch, prm, force_name=model_name
                )
                print(f"Model saved to database with accuracy: {accuracy}")
            else:
                print(
                    f"Model accuracy {accuracy} is below the minimum threshold "
                    f"{minimum_accuracy}. Not saved."
                )
    finally:
        remove(temp_file_path)
        try:
            del train_set
        except NameError:
            pass
        try:
            del test_set
        except NameError:
            pass
        try:
            if trainer:
                del trainer.model
        except NameError:
            pass
        release_memory()

    return model_name, accuracy, accuracy_to_time, res["score"]
