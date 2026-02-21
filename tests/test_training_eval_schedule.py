from __future__ import annotations

import argparse

import pytest

from training.train import _parse_eval_each_epoch_cli, _resolve_val_every


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("False", False),
        ("yes", True),
        ("off", False),
        ("7", 7),
    ],
)
def test_parse_eval_each_epoch_cli(raw: str, expected: int | bool) -> None:
    assert _parse_eval_each_epoch_cli(raw) == expected


def test_parse_eval_each_epoch_cli_rejects_invalid_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_eval_each_epoch_cli("sometimes")


def test_resolve_val_every_with_alias_true() -> None:
    assert _resolve_val_every(val_every=50, eval_each_epoch=True) == 1


def test_resolve_val_every_with_alias_false_keeps_val_every() -> None:
    assert _resolve_val_every(val_every=50, eval_each_epoch=False) == 50


def test_resolve_val_every_with_alias_interval() -> None:
    assert _resolve_val_every(val_every=50, eval_each_epoch=3) == 3


def test_resolve_val_every_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        _resolve_val_every(val_every=50, eval_each_epoch=0)
