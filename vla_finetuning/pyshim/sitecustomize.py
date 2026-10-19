"""Imported at interpreter start by the training and export jobs (jobs/train.sh and jobs/export.sh put this
directory on PYTHONPATH, so dataloader workers get it too).

accelerate 1.14 (the RoboTwin env) keeps DummyOptim / DummyScheduler only in accelerate.utils.deepspeed, while
RoboOrchardLab 241bbc6 imports them from accelerate.utils (robo_orchard_lab/pipeline/training/
hook_based_trainer.py:28; they are used only on the DeepSpeed path). Re-export them there; nothing else.
"""

try:
    import accelerate.utils as _au

    if not hasattr(_au, "DummyOptim"):
        from accelerate.utils.deepspeed import DummyOptim, DummyScheduler

        _au.DummyOptim, _au.DummyScheduler = DummyOptim, DummyScheduler
except Exception as _exc:  # noqa: BLE001 - never break interpreter start-up
    import sys

    print(f"[vla_finetuning shim] {_exc}", file=sys.stderr)
