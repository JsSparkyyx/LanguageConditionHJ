"""TextStore extra file (review O7): a reload keeps earlier extras, concurrent writers keep each other's."""

import torch

from main.lang.text import TextStore

from .test_train import StubText


def _store(path):
    return TextStore((), "holobrain_bert", "cpu", extra_path=path, encoder=StubText())


def _saved(path):
    return set(torch.load(path, weights_only=False)["index"])


def test_extra_file_keeps_loaded_and_concurrent_entries(tmp_path):
    p = tmp_path / "text_cache_extra.pt"
    a = _store(p)
    a.add(["keep the cup still", "do not touch the bowl"])
    a.save_extra()
    assert _saved(p) == {"keep the cup still", "do not touch the bowl"}
    b, c = _store(p), _store(p)                  # two processes load the same file
    b.add(["leave the box where it is"])
    b.save_extra()
    assert _saved(p) == {"keep the cup still", "do not touch the bowl", "leave the box where it is"}
    c.add(["don't spill the beads"])
    c.save_extra()                               # c never saw b's sentence: merged from the file
    assert _saved(p) == {"keep the cup still", "do not touch the bowl", "leave the box where it is",
                         "don't spill the beads"}
    d = _store(p)
    assert len(d) == 4 and torch.equal(d.lookup(["don't spill the beads"])[0], c.lookup(["don't spill the beads"])[0])


def test_save_waits_for_another_saver(tmp_path):
    """B2: a save waits while another process holds the extra file's lock, then merges and writes."""
    import threading
    import time
    from pathlib import Path

    p = tmp_path / "text_cache_extra.pt"
    a = _store(p)
    a.add(["keep the cup still"])
    lock = Path(f"{p}.lock")
    lock.mkdir()                                   # another saver is in its critical section
    t = threading.Thread(target=a.save_extra)
    t.start()
    time.sleep(1.0)
    assert t.is_alive() and not p.exists()         # waiting, nothing written yet
    lock.rmdir()
    t.join(timeout=10)
    assert not t.is_alive() and _saved(p) == {"keep the cup still"} and not lock.exists()
