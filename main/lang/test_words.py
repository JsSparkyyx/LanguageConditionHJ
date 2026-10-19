"""Words never used in a TEST sentence (Yuxuan, 2026-10-19: "don't use rice when you do test").

The vanilla and Qwen round runs train with the pre-10-12 vocabulary (legacy_pool.install_vocab; the Qwen run on
purpose, to match the vanilla runs' words), whose bead material synonyms are beads / beans / rice / grains. The
10-12 rename dropped rice, beans and grains; the evaluators install the checkpoint's own vocabulary, so without this
filter a bead-task test sentence could still say "keep all the rice inside". install() makes
language.sentences / language.sample_sentence skip every sentence with a banned word (whole word, any case), in
the evaluators only (training code never imports this module). "beads" is always among the material words, so no
bead rule loses all its sentences; a rule left with none is simply not scored.
"""

from __future__ import annotations

import re

import numpy as np

BANNED = ("rice", "beans", "grains")
_PAT = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.IGNORECASE)


def ok(sentence: str) -> bool:
    return _PAT.search(sentence or "") is None


def install() -> None:
    from main.item_envs import language

    if getattr(language.sentences, "_test_words", False):
        return
    orig_sentences, orig_sample = language.sentences, language.sample_sentence

    def sentences(rule, scene, split="train", bank=None):
        return [s for s in orig_sentences(rule, scene, split, bank) if ok(s)]

    def sample_sentence(rule, scene, rng: np.random.Generator, split="train", bank=None):
        for _ in range(64):
            s = orig_sample(rule, scene, rng, split, bank)
            if ok(s):
                return s
        allowed = sentences(rule, scene, split, bank)
        if not allowed:
            raise ValueError(f"rule {rule.get('rule_id')}: every {split} sentence has a banned test word {BANNED}")
        return allowed[int(rng.integers(len(allowed)))]

    sentences._test_words = sample_sentence._test_words = True
    language.sentences, language.sample_sentence = sentences, sample_sentence
