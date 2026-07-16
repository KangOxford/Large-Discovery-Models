import unittest

from pdf2markush_workflow import (
    merge_formula_records,
    normalize_formula_with_reason,
    parse_formula,
    parse_selectivity_fold,
)


class SelectivityParsingTests(unittest.TestCase):
    def test_plain_potency_is_not_selectivity(self) -> None:
        self.assertIsNone(parse_selectivity_fold("0.18 uM"))
        self.assertIsNone(parse_selectivity_fold("22 nM"))
        self.assertIsNone(parse_selectivity_fold(""))

    def test_explicit_fold_markers_are_selectivity(self) -> None:
        self.assertEqual(parse_selectivity_fold("0.34 nM (15x)"), 15.0)
        self.assertEqual(parse_selectivity_fold("29-fold over WT"), 29.0)
        self.assertEqual(parse_selectivity_fold("selectivity 7.2"), 7.2)


class FormulaNormalizationTests(unittest.TestCase):
    def test_mh_plus_label_without_formula_charge_is_treated_as_neutral(self) -> None:
        formula, reason = normalize_formula_with_reason("C31H38ClF3N8O3", "mh_plus", "table")
        self.assertEqual(formula, "C31H38ClF3N8O3")
        self.assertEqual(reason, "table_formula_treated_as_neutral")

    def test_explicit_formula_charge_subtracts_hydrogen(self) -> None:
        formula, reason = normalize_formula_with_reason("C10H11N+", "mh_plus", "characterization")
        self.assertEqual(formula, "C10H10N")
        self.assertEqual(reason, "subtracted_h_from_explicit_mh_plus_formula")

    def test_mh_plus_annotation_is_not_parsed_as_elements(self) -> None:
        self.assertEqual(parse_formula("C10H11N [M+H]+"), {"C": 10, "H": 11, "N": 1})
        formula, reason = normalize_formula_with_reason("C10H11N [M+H]+", "mh_plus", "characterization")
        self.assertEqual(formula, "C10H11N")
        self.assertEqual(reason, "mh_plus_label_without_formula_ion_marker_treated_as_neutral")

    def test_merge_formula_records_preserves_normalization_reason(self) -> None:
        merged = merge_formula_records(
            [
                {
                    "tables": [],
                    "records": [
                        {
                            "compound_id": "7",
                            "raw_formula": "C31H38ClF3N8O3",
                            "formula_kind": "mh_plus",
                            "source_type": "table",
                            "page_refs": ["p1"],
                            "evidence": "Table formula with separate [M+H]+ m/z column.",
                            "confidence": 1.0,
                        }
                    ],
                    "notes": [],
                }
            ],
            ["7"],
            [],
        )
        record = merged["records"][0]
        self.assertEqual(record["neutral_formula"], "C31H38ClF3N8O3")
        self.assertEqual(record["formula_normalization_reasons"], ["table_formula_treated_as_neutral"])


if __name__ == "__main__":
    unittest.main()
