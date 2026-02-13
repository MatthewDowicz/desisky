# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for validation split utilities (_splits.py)."""

import numpy as np
import pandas as pd
import pytest

from desisky.data._splits import get_validation_mask


@pytest.fixture
def sample_metadata():
    """DataFrame with EXPID column mimicking VAC metadata."""
    return pd.DataFrame({
        "EXPID": [100, 200, 300, 400, 500],
        "SUNALT": [-25.0, -30.0, -22.0, -28.0, -35.0],
    })


@pytest.fixture
def model_meta_with_val():
    """Model metadata containing val_expids."""
    return {
        "schema": 1,
        "arch": {"in_ch": 1, "out_ch": 1, "meta_dim": 8},
        "training": {
            "sigma_data": 1.0,
            "val_expids": [200, 400, 600],
        },
    }


@pytest.fixture
def model_meta_without_val():
    """Model metadata without val_expids."""
    return {
        "schema": 1,
        "arch": {"in_ch": 1, "out_ch": 1, "meta_dim": 8},
        "training": {"sigma_data": 1.0},
    }


class TestGetValidationMask:
    """Tests for get_validation_mask."""

    def test_returns_bool_ndarray(self, sample_metadata, model_meta_with_val):
        mask = get_validation_mask(sample_metadata, model_meta_with_val)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_

    def test_correct_shape(self, sample_metadata, model_meta_with_val):
        mask = get_validation_mask(sample_metadata, model_meta_with_val)
        assert mask.shape == (len(sample_metadata),)

    def test_matches_correct_expids(self, sample_metadata, model_meta_with_val):
        mask = get_validation_mask(sample_metadata, model_meta_with_val)
        # EXPIDs 200 and 400 are in both metadata and val_expids
        # EXPID 600 is in val_expids but not in metadata (expected for VAC subset)
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_no_matches_returns_all_false(self, sample_metadata):
        meta = {
            "schema": 1,
            "training": {"val_expids": [999, 888, 777]},
        }
        mask = get_validation_mask(sample_metadata, meta)
        assert not mask.any()

    def test_all_match(self):
        metadata = pd.DataFrame({"EXPID": [1, 2, 3]})
        meta = {"training": {"val_expids": [1, 2, 3]}}
        mask = get_validation_mask(metadata, meta)
        assert mask.all()

    def test_missing_val_expids_raises(self, sample_metadata, model_meta_without_val):
        with pytest.raises(ValueError, match="val_expids"):
            get_validation_mask(sample_metadata, model_meta_without_val)

    def test_missing_training_key_raises(self, sample_metadata):
        meta = {"schema": 1, "arch": {}}
        with pytest.raises(ValueError, match="val_expids"):
            get_validation_mask(sample_metadata, meta)

    def test_missing_expid_column_raises(self, model_meta_with_val):
        metadata = pd.DataFrame({"SUNALT": [-25.0, -30.0]})
        with pytest.raises(KeyError, match="EXPID"):
            get_validation_mask(metadata, model_meta_with_val)

    def test_prints_summary(self, sample_metadata, model_meta_with_val, capsys):
        get_validation_mask(sample_metadata, model_meta_with_val)
        captured = capsys.readouterr()
        assert "Found 2 of 3 validation EXPIDs" in captured.out
