import unittest

import cmp_dict


class TestCmpDict(unittest.TestCase):

    def _compare(self, test, reference, tolerance=1e-8):
        return cmp_dict.abs_diff(tolerance)(test, reference)

    def test_matching_nested_values_pass(self):
        data = {"f.bin": {((0, 0, 0, 1, 1), 0, 0): [[1.0, 2.0], [3.0, 4.0]]}}
        passed, msg = self._compare(data, data)
        self.assertTrue(passed)
        self.assertIn("over 4 values", msg)

    def test_difference_within_tolerance_passes(self):
        passed, _ = self._compare({"a": [1.0]}, {"a": [1.0 + 1e-10]})
        self.assertTrue(passed)

    def test_difference_above_tolerance_fails_and_reports_location(self):
        passed, msg = self._compare({"a": {"b": [1.0, 2.0]}}, {"a": {"b": [1.0, 2.1]}})
        self.assertFalse(passed)
        self.assertIn("('a', 'b', 2)", msg)

    def test_complex_values_compare_both_parts(self):
        passed, _ = self._compare({"a": [1 + 1j]}, {"a": [1 + 1.1j]})
        self.assertFalse(passed)
        passed, _ = self._compare({"a": [1 + 1j]}, {"a": [1 + 1j]})
        self.assertTrue(passed)

    def test_key_mismatch_fails_in_both_directions(self):
        for test, reference in (({"a": [1.0], "b": [1.0]}, {"a": [1.0]}),
                                ({"a": [1.0]}, {"a": [1.0], "b": [1.0]})):
            with self.subTest(test=test):
                passed, msg = self._compare(test, reference)
                self.assertFalse(passed)
                self.assertIn("key mismatch", msg)

    def test_shape_mismatch_fails(self):
        passed, msg = self._compare({"a": [1.0, 2.0]}, {"a": [1.0]})
        self.assertFalse(passed)
        self.assertIn("shape mismatch", msg)

    def test_dictionary_against_number_fails(self):
        passed, msg = self._compare({"a": {"b": [1.0]}}, {"a": [1.0]})
        self.assertFalse(passed)
        self.assertIn("mismatch", msg)

    def test_no_values_fails(self):
        for data in ({}, {"a": {}}, {"a": []}):
            with self.subTest(data=data):
                passed, msg = self._compare(data, data)
                self.assertFalse(passed)
                self.assertIn("no numeric values", msg)

    def test_non_dictionary_input_fails(self):
        passed, msg = self._compare([1.0], [1.0])
        self.assertFalse(passed)
        self.assertIn("expected dictionaries", msg)

    def test_matching_nonfinite_values_fail(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                passed, msg = self._compare({"a": [value]}, {"a": [value]})
                self.assertFalse(passed)
                self.assertIn("nonfinite", msg)

    def test_nonfinite_tolerance_is_rejected(self):
        for tolerance in (float("nan"), float("inf"), -1.0):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    cmp_dict.abs_diff(tolerance)


if __name__ == "__main__":
    unittest.main()
