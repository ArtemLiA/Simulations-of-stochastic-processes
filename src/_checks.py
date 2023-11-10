from numpy import sum as nsum
from numpy import isclose


class ArgumentError(ValueError):
    pass


class ArgumentIncorrectValueError(ArgumentError):
    pass


def is_square_matrix(matrix):
    if len(matrix.shape) != 2:
        raise ArgumentIncorrectValueError("Matrix must be a 2-dimensional array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ArgumentIncorrectValueError("Matrix is not square")


def is_stochastic_matrix(matrix):
    if (matrix < 0).any():
        raise ArgumentIncorrectValueError("Matrix is not stochastic: there is negative element")
    if not isclose(nsum(matrix, axis=1), 1).all():
        raise ArgumentIncorrectValueError("Matrix is not stochastic: there is a row with the sum of elements unequal to 1")


def is_stochastic_vector(vector):
    if not (vector >= 0).all():
        raise ArgumentIncorrectValueError("Vector is not stochastic: there is negative element")
    if not isclose(nsum(vector), 1):
        raise ArgumentIncorrectValueError("Vector is not stochastic: sum of elements unequal to 1")


def is_agreed(vector, matrix):
    """
    Check consistency of the dimensions of the vector-row and matrix
    """
    if not vector.shape[0] == matrix.shape[0]:
        raise ArgumentIncorrectValueError("Inconsistency of the dimensions of the row-vector and matrix")
