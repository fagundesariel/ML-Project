"""Shared helpers for building reproducible modeling pipelines."""

from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

DEFAULT_RANDOM_STATE = 42


def make_stratified_cv(
    n_splits=5,
    *,
    shuffle=True,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Return the repository-standard stratified cross-validation splitter."""

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def make_nested_stratified_cv(
    outer_splits=5,
    inner_splits=3,
    *,
    shuffle=True,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Return outer and inner stratified splitters for nested validation."""

    return (
        make_stratified_cv(
            n_splits=outer_splits,
            shuffle=shuffle,
            random_state=random_state,
        ),
        make_stratified_cv(
            n_splits=inner_splits,
            shuffle=shuffle,
            random_state=random_state,
        ),
    )


def make_modeling_pipeline(
    classifier,
    *,
    sampler=None,
    reducer=None,
    scaler=None,
):
    """Build the shared preprocessing and classification pipeline."""

    scaler_step = MinMaxScaler() if scaler is None else scaler
    steps = [("scaler", clone(scaler_step))]

    if sampler is not None:
        steps.append(("sampler", clone(sampler)))

    if reducer is not None:
        steps.append(("reducer", clone(reducer)))

    steps.append(("classifier", clone(classifier)))
    return Pipeline(steps)
