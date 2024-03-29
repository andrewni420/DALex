from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import psb2

from push4.gp.soup import Soup, CoreSoup, GeneToken, CoreSoupEmpty, rand_bool, rand_float, rand_int
from push4.lang.dag import Dag
from push4.lang.expr import Input, Function, Method, Constant
from push4.lang.hof import LocalInput, MapExpr
from push4.lang.push import Push
from push4.lang.reify import RetToElementType, ArgsToElementType
from push4.library.io import do_print, _pass_do, print_do
from push4.library.op import add, _max_numeric, sum_, div, max_
from push4.library.str import String
from push4.library.collections import len_, in_
from push4.utils import damerau_levenshtein_distance


DATA_ROOT = "examples/software_synthesis/data/"
PSB2_ROOT = "../../Data/PSB2"

# The penalty error to assign empty programs and programs which produce runtime errors.
penalty = 1e5


def float_to_empty_str(x):
    if isinstance(x, float):
        return ""
    return x


def escape_str(s: str) -> str:
    ret = ""
    for c in s:
        if c == "\n":
            ret += "\\n"
        else:
            ret += c
    return ret


class Problem(ABC):

    def __init__(self, name: str, output_type: type, arity: int):
        self.name = name
        self.output_type = output_type
        self.arity = arity
        self.arg_names = ["input" + str(i + 1) for i in range(arity)]
        self._training_cases = None
        self._test_cases = None
        self.downsampled_cases=None

    @property
    def training_cases(self):
        if self._training_cases is None:
            self._training_cases = self.read_cases(100)
        return self._training_cases

    def get_cases(self, cases: List[int]):
        if cases is None:
            return self.training_cases
        else:
            return [self.training_cases[i] for i in cases]

    def sample_cases(self, downsample_rate=1.):
        n=len(self.training_cases)
        downsample_size = int(downsample_rate * n)
        rng = np.random.default_rng()
        self.downsampled_cases = rng.choice(n,replace=False,size=downsample_size)

    @property
    def test_cases(self):
        if self._test_cases is None:
            self._test_cases = self.read_cases(1000)
        return self._test_cases

    def read_cases(self, n_random_cases: int) -> List[Dict]:
        edge_case_path = DATA_ROOT + "{n}/{n}-edge.csv.gz".format(n=self.name)
        edge_cases = pd.read_csv(edge_case_path)
        random_case_path = DATA_ROOT + \
            "{n}/{n}-random.csv.gz".format(n=self.name)
        all_random_cases = pd.read_csv(random_case_path)
        random_cases = all_random_cases.sample(n=n_random_cases)
        return pd.concat([edge_cases, random_cases]).to_dict('records')

    def train_error(self, program: Dag, downsampled=True) -> np.array:
        return self.error_fn(program, self.get_cases(self.downsampled_cases)) if downsampled else self.error_fn(program, self.training_cases)

    def test_error(self, program: Dag) -> np.array:
        return self.error_fn(program, self.test_cases)

    @abstractmethod
    def soup(self) -> Soup:
        ...

    @abstractmethod
    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        ...


class Basement(Problem):
    def __init__(self):
        super().__init__("basement", int, 1)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "basement", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", List[int])
            .register_constants([0, 1, -1,[]])
            .register_erc_generator(rand_int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class SubstitutionCipher(Problem):
    def __init__(self):
        super().__init__("substitution-cipher", str, 3)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "substitution-cipher", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", str)
            .register_input("input2", str)
            .register_input("input3", str)
            .register_constants([0, ""])
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class GCD(Problem):
    def __init__(self):
        super().__init__("gcd", int, 2)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "gcd", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", int)
            .register_input("input2", int)
            .register_erc_generator(rand_int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class Twitter(Problem):
    def __init__(self):
        super().__init__("twitter", str, 1)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "twitter", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", str)
            .register_constants([0, 140, "Too many characters", "You didn't type anything", "your tweet has " , " characters"])
            .register_erc_generator(rand_int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class FuelCost(Problem):
    def __init__(self):
        super().__init__("fuel-cost", int, 1)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "fuel-cost", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", List[int])
            .register_constants([0, 1, 2, 3])
            .register_erc_generator(rand_int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class FizzBuzz(Problem):
    def __init__(self):
        super().__init__("fizz-buzz", str, 1)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "fizz-buzz", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", int)
            .register_constants(["Fizz", "Buzz", "FizzBuzz", 0, 3, 5])
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(damerau_levenshtein_distance(y_true, y_pred))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class MiddleCharacter(Problem):
    def __init__(self):
        super().__init__("middle-character", str, 1)
        self._training_cases, self._test_cases = psb2.fetch_examples(
            PSB2_ROOT, "middle-character", 200, 2000)

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", str)
            .register_constants(["", 0, 1, 2])
            .register_erc_generator(rand_int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(damerau_levenshtein_distance(y_true, y_pred))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class CompareStringLengths(Problem):

    def __init__(self):
        super().__init__("compare-string-lengths", bool, 3)

    def soup(self) -> Soup:
        return (
            CoreSoupEmpty()
            .register_input("input1", str)
            .register_input("input2", str)
            .register_input("input3", str)
            .register_erc_generator(rand_bool)
        )

    @property
    def training_cases(self):
        if self._training_cases is None:
            self._training_cases = self.read_cases(1000)
        return self._training_cases

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
                errors.append(penalty)
            else:
                y_true = case["output1"]
                inputs = case.copy()
                del inputs["output1"]
                inputs = {k: float_to_empty_str(v) for k, v in inputs.items()}
                try:
                    y_pred = program.eval(**inputs)
                    errors.append(float(not isinstance(y_pred, bool)))
                    errors.append(float(not (bool(y_pred) == y_true)))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
                    errors.append(penalty)

        return np.array(errors)


# class CompareStringLengths(Problem):

#     def __init__(self):
#         super().__init__("compare-string-lengths", bool, 3)

#     def soup(self) -> Soup:
#         return (
#             CoreSoup()
#             .register_input("input1", str)
#             .register_input("input2", str)
#             .register_input("input3", str)
#         )

#     def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
#         errors = []
#         for case in cases:
#             if program is None:
#                 errors.append(penalty)
#                 errors.append(penalty)
#             else:
#                 y_true = case["output1"]
#                 inputs = case.copy()
#                 del inputs["output1"]
#                 inputs = {k: float_to_empty_str(v) for k, v in inputs.items()}
#                 try:
#                     y_pred = program.eval(**inputs)
#                     errors.append(float(not isinstance(y_pred, bool)))
#                     errors.append(float(not (bool(y_pred) == y_true)))
#                 except Exception as e:
#                     # print(e)
#                     errors.append(penalty)
#                     errors.append(penalty)
#         return np.array(errors)


class Median(Problem):

    def __init__(self):
        super().__init__("median", Any, 3)

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", int)
            .register_input("input2", int)
            .register_input("input3", int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = str(case["output1"])
                inputs = case.copy()
                del inputs["output1"]
                try:
                    program.eval(**inputs)
                    y_pred = program.stdout()
                    errors.append(float(not y_pred == y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class NumberIO(Problem):

    def __init__(self):
        super().__init__("number-io", Any, 2)

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", float)
            .register_input("input2", int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = str(case["output1"])[:10]  # Prevents rounding errors.
                inputs = case.copy()
                del inputs["output1"]
                try:
                    program.eval(**inputs)
                    y_pred = program.stdout()[:10]
                    errors.append(damerau_levenshtein_distance(y_true, y_pred))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class ReplaceSpaceWithNewline(Problem):

    def __init__(self):
        super().__init__("replace-space-with-newline", int, 1)

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", str)
            .register_constants([" ", "\n", ""])
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            case = {k: float_to_empty_str(v) for k, v in case.items()}
            if program is None:
                errors.append(penalty)
                errors.append(penalty)
            else:
                y_true_stdout = case["output1"]
                y_true = case["output2"]
                int_err = penalty
                stdout_err = penalty
                try:
                    y_pred = program.eval(input1=case["input1"])
                    y_pred_stdout = program.stdout()
                    int_err = abs(y_true - y_pred)
                    stdout_err = damerau_levenshtein_distance(
                        y_true_stdout, escape_str(y_pred_stdout))
                except Exception as e:
                    # print(e)
                    ...
                errors.append(int_err)
                errors.append(stdout_err)
        return np.array(errors)


class Smallest(Problem):

    def __init__(self):
        super().__init__("smallest", Any, 4)

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", int)
            .register_input("input2", int)
            .register_input("input3", int)
            .register_input("input4", int)
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = str(case["output1"])
                inputs = case.copy()
                del inputs["output1"]
                try:
                    program.eval(**inputs)
                    y_pred = program.stdout()
                    errors.append(float(not y_pred == y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class VectorAverage(Problem):

    def __init__(self):
        super().__init__("vector-average", float, 1)
        self.parse_cases(self.training_cases)
        self.parse_cases(self.test_cases)

    def parse_cases(self, cases: List[Dict]):
        for case in cases:
            case["input1"] = self._parse_list(case["input1"])

    def _parse_list(self, list_str: str) -> List[float]:
        return [float(s) for s in list_str[1:-1].split(" ")]

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", List[float])
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = round(case["output1"], 4)
                try:
                    y_pred = round(program.eval(input1=case["input1"]), 4)
                    errors.append(abs(y_pred - y_true))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


class NegativeToZero(Problem):

    def __init__(self,):
        super().__init__("negative-to-zero", List[int], 1)
        self.parse_cases(self.training_cases)
        self.parse_cases(self.test_cases)

    def parse_cases(self, cases: List[Dict]):
        for case in cases:
            case["input1"] = self._parse_list(case["input1"])
            case["output1"] = self._parse_list(case["output1"])

    def _parse_list(self, list_str: str) -> List[int]:
        no_braces = list_str[1:-1]
        if no_braces == "":
            return []
        return [int(s) for s in no_braces.split(" ")]

    def soup(self) -> Soup:
        return (
            CoreSoup()
            .register_input("input1", List[int])
        )

    def error_fn(self, program: Dag, cases: List[Dict]) -> np.array:
        errors = []
        for case in cases:
            if program is None:
                errors.append(penalty)
            else:
                y_true = case["output1"]
                try:
                    y_pred = program.eval(input1=case["input1"])
                    errors.append(damerau_levenshtein_distance(y_true, y_pred))
                except Exception as e:
                    # print(e)
                    errors.append(penalty)
        return np.array(errors)


# class StringLengthBackwards(Problem):
#
#     def __init__(self):
#         super().__init__("string-length-backwards", Any, ["lst"])
#
#     def soup(self) -> Soup:
#         return (
#             CoreSoup()
#             .register_input("lst", List[str])
#         )
