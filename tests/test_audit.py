"""Hermetic tests for audit bucketing helpers."""
from __future__ import annotations

from topicsleuth.audit import _evidence_bucket, _strength_bucket


class TestStrengthBucket:
    def test_below_03(self):
        assert _strength_bucket(0.0) == "<0.3"
        assert _strength_bucket(0.29) == "<0.3"

    def test_03_to_05_inclusive_lower(self):
        assert _strength_bucket(0.30) == "0.3-0.5"
        assert _strength_bucket(0.49) == "0.3-0.5"

    def test_05_to_07(self):
        assert _strength_bucket(0.50) == "0.5-0.7"
        assert _strength_bucket(0.69) == "0.5-0.7"

    def test_07_to_08(self):
        assert _strength_bucket(0.70) == "0.7-0.8"
        assert _strength_bucket(0.79) == "0.7-0.8"

    def test_08_to_09(self):
        assert _strength_bucket(0.80) == "0.8-0.9"
        assert _strength_bucket(0.89) == "0.8-0.9"

    def test_09_and_above(self):
        assert _strength_bucket(0.90) == "0.9+"
        assert _strength_bucket(0.95) == "0.9+"
        assert _strength_bucket(1.0) == "0.9+"


class TestEvidenceBucket:
    def test_singletons(self):
        assert _evidence_bucket(0) == "0-1"
        assert _evidence_bucket(1) == "0-1"

    def test_small(self):
        assert _evidence_bucket(2) == "2-4"
        assert _evidence_bucket(4) == "2-4"

    def test_medium(self):
        assert _evidence_bucket(5) == "5-9"
        assert _evidence_bucket(9) == "5-9"

    def test_growing(self):
        assert _evidence_bucket(10) == "10-19"
        assert _evidence_bucket(19) == "10-19"

    def test_mature(self):
        assert _evidence_bucket(20) == "20-29"
        assert _evidence_bucket(29) == "20-29"

    def test_evidence_rich(self):
        assert _evidence_bucket(30) == "30+"
        assert _evidence_bucket(100) == "30+"
        assert _evidence_bucket(1000) == "30+"
