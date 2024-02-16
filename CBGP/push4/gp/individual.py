"""The :mod:`individual` module defines an Individaul in an evolutionary population.

Individuals are made up of Genomes, which are the linear Push program
representations which can be manipulated by seach algorithms.

"""
from typing import Union, Sequence

import numpy as np
from pyrsistent import pvector

from push4.gp.spawn import genome_to_push_code
from push4.lang.dag import Dag
from push4.lang.expr import Expression
from push4.lang.push import Push


Genome = Sequence[Expression]


class Individual:
    """An individual in an evolutionary population.

    Attributes
    ----------
    genome : Genome
        The Genome of the Individual.
    error_vector : np.array
        An array of error values produced by evaluating the Individual's program.
    total_error : float
        The sum of all error values in the Individual's error_vector.
    error_vector_bytes:
        Hashable Byte representation of the individual's error vector.

    """

    __slots__ = [
        "genome", "signature", "output_type", "error_vector", "_inherit_error_bytes", "ind_id", "parent_id", "true_error",
        "_push_code", "_program", "_total_error", "_error_vector_bytes", "inherited_errors", "pprob", "lprob", "wprob"
    ]

    def __init__(self, genome: Genome, output_type: type, inherited_errors = None, ind_id=None, parent_id=None):
        self.output_type = output_type
        self.genome = pvector(genome)
        self._push_code = None
        self._program = None
        self.error_vector = None
        self._total_error = None
        self._error_vector_bytes = None
        self._inherit_error_bytes = None
        self.ind_id=ind_id
        self.parent_id=parent_id
        self.pprob=0 
        self.lprob=0
        self.wprob=0
        self.true_error=None
        self.inherited_errors = inherited_errors if inherited_errors is not None else np.array([])

    @property
    def push_code(self):
        if self._push_code is None:
            self._push_code = genome_to_push_code(self.genome)
        return self._push_code

    @property
    def program(self) -> Dag:
        """Push program of individual. Taken from Plush genome."""
        if self._program is None:
            dag = Push().compile(self.push_code, self.output_type)
            self._program = dag
        return self._program

    @property
    def total_error(self) -> Union[np.int64, np.float64]:
        """Numeric sum of the error vector."""
        if self._total_error is None:
            try:
                self._total_error = np.sum(self.error_vector)
            except OverflowError:
                self._total_error = np.inf
        return self._total_error

    @total_error.setter
    def total_error(self, value: Union[np.int64, np.float64]):
        raise AttributeError("Cannot set total_error directly. Must set error_vector.")

    @property
    def error_vector_bytes(self):
        """Hashable Byte representation of the individual's error vector."""
        if self._error_vector_bytes is None:
            self._error_vector_bytes = self.error_vector.data.tobytes()
        return self._error_vector_bytes

    @property
    def inherit_error_bytes(self):
        """Hashable Byte representation of the individual's error vector."""
        if self._inherit_error_bytes is None:
            self._inherit_error_bytes = self.inherited_errors.data.tobytes()
        return self._inherit_error_bytes

    @error_vector_bytes.setter
    def error_vector_bytes(self, value):
        raise AttributeError("Cannot set error_vector_bytes directly. Must set error_vector.")

    def __lt__(self, other):
        return self.total_error < other.total_error

    def __eq__(self, other):
        return isinstance(other, Individual) and self.genome == other.genome

    def json(self):
        return {"ind_id": self.ind_id, 
        "parent_id": self.parent_id, 
        "error_vector": self.error_vector.tolist(),
        "total_error": float(self.total_error),
        "plexi_probability": self.pprob,
        "lexi_probability": self.lprob,
        "wlexi_probability": self.wprob}
