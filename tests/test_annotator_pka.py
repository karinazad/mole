import os
import pandas as pd
import unittest
import numpy as np
from mole.data.annotator_pka import AnnotatorpKa, DEFAULT_PATH_TO_PKA_TABLE, DEFAULT_PATH_TO_PKA_TEST
import logging

logging.disable(logging.CRITICAL)


ROOT_PATH = ""


class TestAnnotatorpKaTable(unittest.TestCase):
    """
    Tests the AnnotatorpKa from mole.data.annotator_pka using a saved table with 3 examples per each
    functional group.
    """

    def setUp(self) -> None:
        self.path_to_table = os.path.join(ROOT_PATH, DEFAULT_PATH_TO_PKA_TABLE)
        self.path_to_test = os.path.join(ROOT_PATH, DEFAULT_PATH_TO_PKA_TEST)

    def test_smarts_annotation(self):
        annot = AnnotatorpKa(self.path_to_table)
        test_df = pd.read_csv(self.path_to_test, index_col="entry")

        groups_check = {}
        groups_detected = {}

        for i in range(len(test_df)):
            group_df = test_df.iloc[i]
            group = group_df["group"]
            examples = group_df[["Example1", "Example2", "Example3"]]
            res = annot.annotate(examples)

            detected = res.groupby("ID").apply(lambda x: x["group"].values)
            group_present = res.groupby("ID").apply(lambda x: group in x["group"].values)

            groups_detected[group] = list(detected)
            groups_check[group] = group_present.values

        df_present = pd.DataFrame(groups_check,
                                  index=["Example1: Desired Group Present?", "Example2: Desired Group Present?",
                                         "Example3: Desired Group Present?"]).T
        df_detected = pd.DataFrame(groups_detected, index=["Example1: Groups Detected", "Example2: Groups Detected",
                                                           "Example3: Groups Detected"]).T

        result = test_df.set_index("group")
        result = result.join(df_present)
        result = result.join(df_detected)
        result = result.reset_index()

        result.to_csv("tests/test_smarts.csv")

        ordered_cols = ['group', 'smarts'] + [c for c in sorted(result.columns) if c not in ['group', 'smarts']]
        result = result.reindex(columns=ordered_cols)

        for i in result.index:
            row = result.iloc[i]
            group_name = row["group"]
            present = row[["Example1: Desired Group Present?", "Example2: Desired Group Present?",
                           "Example3: Desired Group Present?"]]
            self.assertTrue(present.all()), f"Failed to identify the following group: {group_name}: {row.to_dict()}"


class TestAnnotatorpKa(unittest.TestCase):
    """
        Tests the AnnotatorpKa from mole.data.annotator_pka using dummy data.
    """

    def setUp(self) -> None:
        self.path_to_table = os.path.join(ROOT_PATH, DEFAULT_PATH_TO_PKA_TABLE)

        self.smiles = pd.Series(
            [
                'COc1ccc([C@@H](O)C[C@H]2c3cc(OC)c(OC)cc3CCN2C)cc1',
                'Cc1nnc(N2CC[C@@H](F)C2)c2nn(-c3ccc(OCC(F)(F)F)cc3)c(C)c12',
                'O=c1n(Cc2ccccc2)c2sc3c(c2c2ncnn12)CCN(CC1CCOCC1)C3',
                'COCCCc1cc(CN(C(=O)[C@H]2CNCC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(OCC(C)(C)C(=O)O)c1',
                'CC1(C)C2=C3C=C4C5=[N+](CCC4OC3CCN2c2ccc(CC(=O)NCCCN(CCOc3ccc(NS(C)(=O)=O)cc3)CCc3ccc(NS(C)(=O)=O)cc3)cc21)c1ccc(S(=O)(=O)[O-])cc1C5(C)C',
                'Fc1ccc(-c2c[nH]c([C@@H]3CCc4[nH]c5ccccc5c4C3)n2)cc1',
                'Fc1ccc(-c2c[nH]c([C@H]3Cc4c([nH]c5ccccc45)CN3)n2)cc1',
                'COc1ccc2ncc(C(F)(F)F)c(CCC34CCC(NCc5ccc6c(n5)NC(=O)CO6)(CC3)CO4)c2n1',
                'Cc1ccccc1CN1[C@H]2CC[C@@H]1C[C@@H](Oc1cccc(C(N)=O)c1)C2',
                'O=[N+]([O-])c1cccc(CNc2cc(C(F)(F)F)cc3ncc(N4CCN(CCO)CC4)cc23)c1',
                'CCCC',
                'NC'
            ])
        self.y = pd.Series([
            4.4, 5.1, 150.5, 11560, 0, 3.4, 5, 100000, 5, -5, -15000, 4])

    def test(self) -> None:
        annotator = AnnotatorpKa(self.path_to_table)
        df = annotator.annotate(self.smiles)

        self.assertEqual({'base', 'no match', 'zwitterion'}, set(df["Compound Annotation"].unique()))

    def test_y(self) -> None:
        annotator = AnnotatorpKa(self.path_to_table)
        df = annotator.annotate(self.smiles, self.y)
        df["experimental pka"] = self.y

        self.assertEqual({'in range', 'out of range', 'undetermined'}, set(df["Experimental pKa in range"].unique()))

        matched_df = df[~(df["Compound Annotation"] == "no match")]
        mask_within_range = (matched_df["experimental pka"].astype(float) > matched_df["lower_limit"]) \
                            & (matched_df["experimental pka"].astype(float) < matched_df["upper_limit"])

        # self.assertEqual(len(matched_df[mask_within_range.to_numpy()]), 8)

        zwitterion_out_df = df[df["Compound Annotation"] == "zwitterion (out of range)"]
        self.assertTrue(all([x == "out of range" for x in zwitterion_out_df["Experimental pKa in range"]]))

    def test_y_limit(self) -> None:
        annotator1 = AnnotatorpKa(self.path_to_table, use_extended_std=True)
        df1 = annotator1.annotate(self.smiles, self.y)
        matched_df1 = df1[df1["group"] != "no match"]

        annotator2 = AnnotatorpKa(self.path_to_table, use_extended_std=False)
        df2 = annotator2.annotate(self.smiles, self.y)
        matched_df2 = df2[df2["group"] != "no match"]

        self.assertTrue(all(list(matched_df1["sd"] < matched_df1["extended_sd"])))
        self.assertTrue(all(list(matched_df1["lower_limit"] < matched_df2["lower_limit"])))
        self.assertTrue(all(list(matched_df1["upper_limit"] > matched_df2["upper_limit"])))

    def test_group_annotation(self) -> None:
        annotator = AnnotatorpKa(self.path_to_table)

        # Strong acid
        df = pd.DataFrame({
            "pka_subclass": ["weak_acid", "weak_base", "strong_acid"],
            "pka_class": ["acid", "base", "acid"],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "acid")

        # Zwitterion
        df = pd.DataFrame({
            "pka_subclass": ["strong_base", "weak_base", "strong_acid"],
            "pka_class": ["base", "base", "acid"],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "zwitterion")

        # Weak base
        df = pd.DataFrame({
            "pka_subclass": ["weak_base", "weak_base", "weak_base"],
            "pka_class": ["base", "base", "base"],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "base")

        # All weak groups and no experimental pka
        df = pd.DataFrame({
            "pka_subclass": ["weak_base", "weak_base", "weak_acid", "neutral"],
            "pka_class": ["base", "base", "acid", "neutral"],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "zwitterion")

        # All weak groups and no experimental pka
        df = pd.DataFrame({
            "pka_subclass": ['weak_acid', 'weak_acid', 'weak_acid', 'weak_acid', 'weak_base', 'weak_base', 'weak_base',
                             'weak_acid'],
            "pka_class": ['acid', 'acid', 'acid', 'acid', 'base', 'base', 'base', 'acid'],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "zwitterion")

        # All weak groups and experimental pka
        df = pd.DataFrame({
            "pka_subclass": ['weak_acid', 'weak_acid', 'weak_acid', 'weak_acid', 'weak_base', 'weak_base', 'weak_base',
                             'weak_acid'],
            "pka_class": ['acid', 'acid', 'acid', 'acid', 'base', 'base', 'base', 'acid'],
            "lower_limit": [11.233919649, 11.233919649, 11.233919649, 11.233919649, 0.2821368580000003,
                            0.2821368580000003, 0.2821368580000003, 7.133976045],
            "upper_limit": [18.326849591, 18.326849591, 18.326849591, 18.326849591, 8.42495139, 8.42495139, 8.42495139,
                            17.026023955],
            "exp_pka": [3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "base")

        # All weak groups and experimental pka & out of range
        df = pd.DataFrame({
            "pka_subclass": ['weak_acid', 'weak_acid', 'weak_acid', 'weak_acid', 'weak_base', 'weak_base', 'weak_base',
                             'weak_acid'],
            "pka_class": ['acid', 'acid', 'acid', 'acid', 'base', 'base', 'base', 'acid'],
            "lower_limit": [11.233919649, 11.233919649, 11.233919649, 11.233919649, 0.2821368580000003,
                            0.2821368580000003, 0.2821368580000003, 7.133976045],
            "upper_limit": [18.326849591, 18.326849591, 18.326849591, 18.326849591, 8.42495139, 8.42495139, 8.42495139,
                            17.026023955],
            "exp_pka": [3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7, 3.7],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "base")

        # All weak groups and experimental pka & overlap between groups (picks the most common of the matched)
        df = pd.DataFrame({
            "pka_subclass": ['weak_acid', 'weak_acid', 'weak_acid', 'weak_acid', 'weak_base', 'weak_base', 'weak_base',
                             'weak_acid'],
            "pka_class": ['acid', 'acid', 'acid', 'acid', 'base', 'base', 'base', 'acid'],
            "lower_limit": [11.233919649, 11.233919649, 11.233919649, 11.233919649, 0.2821368580000003,
                            0.2821368580000003, 0.2821368580000003, 7.133976045],
            "upper_limit": [18.326849591, 18.326849591, 18.326849591, 18.326849591, 8.42495139, 8.42495139, 8.42495139,
                            17.026023955],
            "exp_pka": [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5],
        })
        group = annotator._get_pka_class_annotation(df)
        self.assertEqual(group, "base")

    def test_range_check(self):
        annotator = AnnotatorpKa(self.path_to_table)
        df = pd.DataFrame({
            "lower_limit": [11.233919649, 11.233919649, 11.233919649, 11.233919649, 0.2821368580000003,
                            0.2821368580000003, 0.2821368580000003, 7.133976045],
            "upper_limit": [18.326849591, 18.326849591, 18.326849591, 18.326849591, 8.42495139, 8.42495139, 8.42495139,
                            17.026023955],
            "exp_pka": 10000 * np.array([7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]),
        })
        in_range = annotator._check_in_range(df)
        self.assertEqual(in_range, "out of range")

        df = pd.DataFrame({
            "lower_limit": [11.233919649, 11.233919649, 11.233919649, 11.233919649, 0.2821368580000003,
                            0.2821368580000003, 0.2821368580000003, 7.133976045],
            "upper_limit": [18.326849591, 18.326849591, 18.326849591, 18.326849591, 8.42495139, 8.42495139, 8.42495139,
                            17.026023955],
            "exp_pka": np.array([7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]),
        })
        in_range = annotator._check_in_range(df)
        self.assertEqual(in_range, "in range")
