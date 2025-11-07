# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Dataset and DataLoader utilities for sky brightness training."""

from __future__ import annotations
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SkyBrightnessDataset(Dataset):
    """
    PyTorch Dataset for DESI sky metadata + flux + 4-band targets (V, g, r, z).

    This dataset provides inputs (metadata features), targets (sky magnitudes),
    and optionally the full flux spectra for each observation.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing observation metadata with columns for input features
        and target magnitudes (SKY_MAG_V_SPEC, SKY_MAG_G_SPEC, etc.).
    flux : np.ndarray
        2D array of flux values. Shape: (n_spectra, n_wavelengths).
    input_features : list[str]
        List of column names from metadata to use as model inputs.
    transform : Callable | None, default None
        Optional transform function to apply to (inputs, targets, spectrum).

    Attributes
    ----------
    metadata : pd.DataFrame
        Observation metadata with reset index.
    flux : np.ndarray
        Flux spectra array.
    input_features : list[str]
        Input feature column names.
    targets : np.ndarray
        Target magnitudes array of shape (N, 4) ordered as V, g, r, z.

    Examples
    --------
    >>> from desisky.data import SkySpecVAC
    >>> vac = SkySpecVAC(download=True)
    >>> wave, flux, meta = vac.load_moon_contaminated()
    >>> input_features = ['MOONSEP', 'MOONFRAC', 'MOONALT', 'OBSALT',
    ...                   'TRANSPARENCY_GFA', 'ECLIPSE_FRAC']
    >>> dataset = SkyBrightnessDataset(meta, flux, input_features)
    >>> inputs, targets, spectrum = dataset[0]
    """

    _BAND_ORDER = ["V", "g", "r", "z"]
    _TARGET_COLS = ["SKY_MAG_V_SPEC", "SKY_MAG_G_SPEC", "SKY_MAG_R_SPEC", "SKY_MAG_Z_SPEC"]

    def __init__(
        self,
        metadata: pd.DataFrame,
        flux: np.ndarray,
        input_features: list[str],
        transform: Optional[Callable] = None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.flux = flux
        self.input_features = input_features
        self.transform = transform

        # Validate input features exist in metadata
        missing_features = set(input_features) - set(metadata.columns)
        if missing_features:
            raise ValueError(f"Input features not found in metadata: {missing_features}")

        # Validate target columns exist
        missing_targets = set(self._TARGET_COLS) - set(metadata.columns)
        if missing_targets:
            raise ValueError(f"Target columns not found in metadata: {missing_targets}")

        # targets shape: (N, 4) ordered V, g, r, z
        self.targets = self.metadata[self._TARGET_COLS].to_numpy(dtype="float32")

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int | list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a single observation or batch of observations.

        Parameters
        ----------
        idx : int | list[int]
            Index or indices of observations to retrieve.

        Returns
        -------
        inputs : np.ndarray
            Input features. Shape: (n_features,) or (batch_size, n_features).
        targets : np.ndarray
            Target magnitudes. Shape: (4,) or (batch_size, 4).
        spectrum : np.ndarray
            Flux spectrum. Shape: (n_wavelengths,) or (batch_size, n_wavelengths).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.metadata.iloc[idx][self.input_features].to_numpy(dtype="float32")
        target = self.targets[idx]  # shape (4,) or (batch, 4)
        spectrum = self.flux[idx]

        if self.transform:
            inputs, target, spectrum = self.transform(inputs, target, spectrum)

        return np.array(inputs), np.array(target), np.array(spectrum)


def numpy_collate(batch: list[Any]) -> list[np.ndarray] | np.ndarray:
    """
    Collate function for PyTorch DataLoader that returns NumPy arrays.

    This allows using JAX arrays downstream while leveraging PyTorch's
    DataLoader for batching and shuffling.

    Parameters
    ----------
    batch : list[Any]
        List of samples from the dataset.

    Returns
    -------
    np.ndarray | list[np.ndarray]
        Collated batch as NumPy array(s).
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    """
    Custom PyTorch DataLoader that returns NumPy arrays instead of tensors.

    This is useful for JAX-based training loops that expect NumPy/JAX arrays.

    Parameters
    ----------
    dataset : Dataset
        PyTorch dataset to load from.
    batch_size : int, default 1
        Number of samples per batch.
    shuffle : bool, default False
        Whether to shuffle data at every epoch.
    num_workers : int, default 0
        Number of subprocesses for data loading.
    **kwargs
        Additional arguments passed to torch.utils.data.DataLoader.

    Examples
    --------
    >>> loader = NumpyLoader(dataset, batch_size=32, shuffle=True)
    >>> for inputs, targets, spectra in loader:
    ...     # inputs, targets, spectra are NumPy arrays
    ...     pass
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        generator: Any = None,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )


def gather_full_data(
    dataloader: NumpyLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Gather all samples from a NumpyLoader and its underlying dataset.

    This is useful for evaluation, where you want to compute metrics on the
    entire dataset (or a subset like the test set) at once.

    Parameters
    ----------
    dataloader : NumpyLoader
        A NumpyLoader wrapping either a SkyBrightnessDataset or a Subset
        of that dataset.

    Returns
    -------
    all_inputs : np.ndarray
        Array of shape (N, num_features) containing all input features.
    all_targets : np.ndarray
        Array of shape (N, 4) containing all target magnitudes.
    all_spectra : np.ndarray
        Array of shape (N, n_wavelengths) containing all flux spectra.
    meta_df : pd.DataFrame
        DataFrame containing the metadata for these N samples.
        Rows are aligned with all_inputs/all_targets/all_spectra.

    Examples
    --------
    >>> inputs, targets, spectra, meta = gather_full_data(test_loader)
    >>> # Compute statistics on the full test set
    >>> mean_targets = targets.mean(axis=0)
    """
    ds = dataloader.dataset

    # Handle Subset (from train_test_split)
    if isinstance(ds, Subset):
        base_dataset = ds.dataset
        indices = ds.indices
    else:
        base_dataset = ds
        indices = range(len(base_dataset))

    # Gather metadata rows
    meta_df = base_dataset.metadata.iloc[indices].reset_index(drop=True)

    # Gather all inputs/targets/spectra
    inputs_list = []
    targets_list = []
    spectra_list = []

    for idx in indices:
        inp, targ, spec = base_dataset[idx]
        inputs_list.append(inp)
        targets_list.append(targ)
        spectra_list.append(spec)

    all_inputs = np.stack(inputs_list, axis=0)
    all_targets = np.array(targets_list)
    all_spectra = np.stack(spectra_list, axis=0)

    return all_inputs, all_targets, all_spectra, meta_df
