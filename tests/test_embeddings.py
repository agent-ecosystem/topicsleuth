"""Hermetic unit tests for topicsleuth.embeddings.

Covers the threshold-crossing detector and the vector helpers used by
refresh_topic_embedding. The MongoDB/Voyage-bound parts of the module are
exercised via integration runs, not here.
"""
from __future__ import annotations

import numpy as np

from topicsleuth.embeddings import (
    CENTROID_WEIGHT,
    TOPIC_EMBEDDING_REFRESH_THRESHOLDS,
    _l2_normalize,
    _normalized_mean,
    crossed_refresh_threshold,
)


class TestCrossedRefreshThreshold:
    def test_zero_to_above_first_threshold(self):
        assert crossed_refresh_threshold(0, 6) is True

    def test_just_below_to_at_threshold(self):
        # 4 -> 5: 5 IS the threshold; condition is prev < T <= new.
        assert crossed_refresh_threshold(4, 5) is True

    def test_at_threshold_to_above_no_new_crossing(self):
        # 5 -> 7: already at 5, hasn't crossed any new threshold yet.
        assert crossed_refresh_threshold(5, 7) is False

    def test_jump_across_multiple_thresholds(self):
        # 13 -> 31 crosses both 15 and 30.
        assert crossed_refresh_threshold(13, 31) is True

    def test_above_threshold_to_above(self):
        # 30 -> 35: above 30, not yet at 50.
        assert crossed_refresh_threshold(30, 35) is False

    def test_crosses_50_threshold(self):
        assert crossed_refresh_threshold(48, 51) is True

    def test_crosses_100_threshold(self):
        assert crossed_refresh_threshold(95, 105) is True

    def test_above_100_no_more_thresholds(self):
        assert crossed_refresh_threshold(101, 110) is False

    def test_decreasing_count_no_refresh(self):
        # Defensive: this shouldn't normally happen, but if a snapshot
        # somehow shows fewer observations than the prior one, no refresh.
        assert crossed_refresh_threshold(10, 5) is False

    def test_no_change_no_refresh(self):
        assert crossed_refresh_threshold(10, 10) is False

    def test_thresholds_are_sorted(self):
        # Sanity check: the constant should remain in ascending order so
        # the crossing detector behavior matches its docstring.
        assert TOPIC_EMBEDDING_REFRESH_THRESHOLDS == sorted(
            TOPIC_EMBEDDING_REFRESH_THRESHOLDS
        )

    def test_centroid_weight_in_valid_range(self):
        # Sanity: blend must remain a convex combination.
        assert 0.0 <= CENTROID_WEIGHT <= 1.0


class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(_l2_normalize(v), v)

    def test_normalizes_to_unit_length(self):
        v = np.array([3.0, 4.0, 0.0])
        result = _l2_normalize(v)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-6

    def test_scaled_unit_vector_normalizes(self):
        v = np.array([0.0, 5.0, 0.0])
        result = _l2_normalize(v)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_zero_vector_returns_zero(self):
        # No division by zero; the helper returns the input unchanged.
        v = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(_l2_normalize(v), v)

    def test_negative_values_handled(self):
        v = np.array([-3.0, 4.0])
        result = _l2_normalize(v)
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-6
        np.testing.assert_array_almost_equal(result, [-0.6, 0.8])


class TestNormalizedMean:
    def test_centroid_of_orthogonal_unit_vectors(self):
        # Mean of [1,0] and [0,1] is [0.5, 0.5]; normalized is [1/sqrt(2)]*2.
        result = _normalized_mean([[1.0, 0.0], [0.0, 1.0]])
        assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-6
        expected = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        np.testing.assert_array_almost_equal(result, expected)

    def test_centroid_of_identical_vectors_unchanged(self):
        v = [1.0, 0.0, 0.0]
        result = _normalized_mean([v, v, v])
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_single_vector_normalized(self):
        result = _normalized_mean([[3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, [0.6, 0.8])
