"""Run with python -m unittest -v. Uses only frozen local data."""

import unittest
from collections import Counter

from build import ROOT, rates, read_csv, validate_inputs, word_counts
from iat_protocol import NEGATIVE, POSITIVE, WORDS


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.responses, cls.images, cls.pairs = validate_inputs()
        cls.counts = word_counts(cls.responses)

    def test_all_words_and_valence_boundary(self):
        self.assertEqual(len(self.counts), 61)
        self.assertEqual({r["word"] for r in self.counts}, set(WORDS))
        self.assertEqual({r["word"] for r in self.counts[:30]}, set(POSITIVE))
        self.assertEqual({r["word"] for r in self.counts[30:]}, set(NEGATIVE))

    def test_descending_pooled_counts_and_ties(self):
        for part in (self.counts[:30], self.counts[30:]):
            keys = [(-r["pooled_count"], WORDS.index(r["word"])) for r in part]
            self.assertEqual(keys, sorted(keys))

    def test_word_totals_and_independent_count(self):
        for race in ("Black", "White"):
            observed = Counter()
            for row in self.responses:
                if row["race"] == race:
                    observed.update(row["words"].split("|"))
            self.assertEqual(sum(observed.values()), 900)
            for row in self.counts:
                self.assertEqual(row[race.lower() + "_count"], observed[row["word"]])
        self.assertEqual(sum(r["black_count"] for r in self.counts[30:]), 39)
        self.assertEqual(sum(r["white_count"] for r in self.counts[30:]), 36)

    def test_pooled_and_standardized_are_distinct(self):
        rows = rates(self.responses)
        for race in ("Black", "White"):
            pooled = next(r["negative_rate_percent"] for r in rows
                          if r["race"] == race and r["expression_group"] == "all_pooled")
            weighted = sum(r["negative_rate_percent"] for r in rows
                           if r["race"] == race and r["expression_group"] != "all_pooled") / 3
            recorded = read_csv(ROOT / "reference/recorded_base_metrics.csv")[0]
            self.assertAlmostEqual(weighted, float(recorded[race.lower() + "_rate"]) * 100)
            self.assertNotAlmostEqual(pooled, weighted)

    def test_saved_sorted_table(self):
        saved = read_csv(ROOT / "results/sorted_word_counts.csv")
        expected = [{k: str(v) for k, v in row.items()} for row in self.counts]
        self.assertEqual(saved, expected)


if __name__ == "__main__":
    unittest.main()
