# This implementation is based on:
# 
# Leo Dorst & Steven De Keninck  
# A Guided Tour to the Plane-Based Geometric Algebra PGA  
# 2022, version 2.0  
# Available at http://www.geometricalgebra.net  
# and http://bivector.net/PGA4CS.html.
# 
# Course notes Geometric Algebra for Computer Graphics  
# SIGGRAPH 2019  
# Charles G. Gunn, Ph. D.
# 
# Geometric Algebra for Computer Science  
# An Object Oriented Approach to Geometry  
# Leo Dorst, Daniel Fontijne, Stephen Mann

import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # type: ignore
import matplotlib.colors as colors
from typing import Literal, cast, Union, Generic, TypeVar, Self, ClassVar, overload, Optional, Final




# index into multivector's data
_Index = int

# basis multivector represented as a tuple
_TBasis = tuple[int, ...]

_Sign = Literal[-1, 0, 1]
# what should basis vectors square to?
_Squares = tuple[_Sign, ...]

# basis multivectors as a tuple
def _basis(n: int) -> tuple[_TBasis, ...]:
    r : tuple[_TBasis, ...] = ((),)
    for e in range(n):
        r += tuple(tail + (e,) for tail in r)
    return r

def _simplify_tbasis(ns: _TBasis, *, squares: _Squares) -> tuple[_Sign, _TBasis]:
    """
    uses the laws
    eN * eN = squares[N]
    eN * eM = - eM * eN  where N != M
    where N, M : int

    examples:
    e1 * e2 * e1 ==> - squares[1] * e2
    e2 * e1 ==> - e1 * e2
    e1 * e2 * e0 * e1 ==> - squares[1] * e0 * e1
    """
    sign: _Sign = 1
    r: list[int] = []
    for e in ns[::-1]:
        for i in range(len(r)):
            if r[i] == e:
                sign = cast(_Sign, sign * squares[e])
                r.remove(e)
                break
            elif r[i] > e: 
                r.insert(i, e)
                break
            else:
                sign = cast(_Sign, -sign)
        else:
            r.append(e)
    return sign, tuple(r)


# tuple to index
def _to_ix(ns: _TBasis, *, squares: _Squares) -> tuple[_Sign, _Index]:
    sign, ns = _simplify_tbasis(ns, squares = squares)
    return sign, sum(1 << n for n in ns)


# precompute multiplication tables for * and ^
def _mul_table(squares: tuple[_Sign, ...]) -> list[tuple[_Index, Literal[1, -1], _Index, _Index]]:
    result = []
    for a in _basis(len(squares)):
        _, a_ix = _to_ix(a, squares = squares)
        for b in _basis(len(squares)):
            _, b_ix = _to_ix(b, squares = squares)
            sign, ab_ix = _to_ix(a + b, squares = squares)
            if sign != 0:
                result.append((ab_ix, sign, a_ix, b_ix))
    return result
# performs | on TBasis
def _contraction_mult(a: _TBasis, b: _TBasis, squares: tuple[_Sign, ...]) -> tuple[_Sign, _TBasis]:
    sign: _Sign = 1
    r = list(b)
    for e in a[::-1]:
        for i in range(len(r)):
            if r[i] == e:
                sign = cast(_Sign, sign * squares[e])
                r.remove(e)
                break
            elif r[i] > e:
                sign = 0
                break
            else:
                sign = cast(_Sign, -sign)
        else:
            sign = 0
    return sign, tuple(r)
# precompute multiplication table for |
def _contraction_table(squares: tuple[_Sign, ...]) -> list[tuple[_Index, Literal[1, -1], _Index, _Index]]:
    result = []
    for a in _basis(len(squares)):
        _, a_ix = _to_ix(a, squares = squares)
        for b in _basis(len(squares)):
            _, b_ix = _to_ix(b, squares = squares)
            sign, r = _contraction_mult(a, b, squares = squares)
            _, r2 = _to_ix(r, squares = squares)
            if sign != 0:
                result.append((r2, sign, a_ix, b_ix))
    return result
_0: _Sign = 0
_1: _Sign = 1



class GeoBase:
    """
    A base class for PGA numbers
    Do not use this class directly
    Use Geo2d for 2D PGA
    Use Geo3d for 3D PGA
    Use Geo if you want to be generic over the number of dimensions

    Overloaded operators:
    - A ^ B -> meet
    - A & B -> join
    - A | B -> left contraction
    """
    
    # I precompite all the operations because _simplify_tbasus() is slow
    _dim: ClassVar[int]
    _basis_count: ClassVar[int]
    _squares: ClassVar[tuple[_Sign, ...]]
    _geometric_mul: ClassVar[list[tuple[int, Literal[1, -1], int, int]]]
    _wedge_mul: ClassVar[list[tuple[int, Literal[1, -1], int, int]]]
    _contraction_mul: ClassVar[list[tuple[int, Literal[1, -1], int, int]]]
    _reverse: ClassVar[NDArray[np.float64]]
    _dual: ClassVar[NDArray[np.float64]]
    _undual: ClassVar[NDArray[np.float64]]
    """
    The origin
    O = e12…n = e0.dual()
    where n is the number of dimensions
    """
    O : ClassVar[Self]
    """
    The pseudoscalar
    I = e01…n
    where n is the number of dimensions
    """
    I : ClassVar[Self]
    """
    The basis vECTors that span the space
    """
    BASIS_VECTORS : ClassVar[tuple[Self, ...]]
    """
    The basis multivectors that span the space
    """
    BASIS_MULTIVECTORS : ClassVar[tuple[Self, ...]]
    def __init_subclass__(cls, *, dimension: Literal[2, 3]) -> None:
        '''
        to create a subclass for a particular dimension use:
          class Geo3d(GeoBase, dimension = 3): pass
        unfortunately exp(), log(), and draw() are
        not supported for dimensions other than 2 and 3
        this is a limitation of this library, not PGA itself
        '''
        dim = dimension + 1
        basis_count = 2**dim
        squares = (_0,) + (_1,)*(dim - 1)
        cls._dim = dim
        cls._basis_count = basis_count
        cls._squares = squares
        cls._geometric_mul = _mul_table(squares)
        cls._wedge_mul = _mul_table(squares = (_0,)*dim)
        cls._contraction_mul = _contraction_table(squares)

        reverse = np.ones(basis_count)
        for i in range(basis_count):
            #       + + - - + + - - ...
            # grade 0 1 2 3 ...
            if i.bit_count() % 4 >= 2:
                reverse[i] = -1
        cls._reverse = reverse

        I_tuple = tuple(range(dim))
        
        dual = np.ones(basis_count)
        for i, e in enumerate(_basis(dim)):
            sign, _ = _to_ix(e[::-1] + I_tuple, squares = (_1,)*dim)
            dual[i] = sign
        cls._dual = dual

        undual = np.ones(basis_count)
        for i, e in enumerate(_basis(dim)):
            sign, _ = _to_ix(I_tuple + e[::-1], squares = (_1,)*dim)
            undual[i] = sign
            pass
        cls._undual = undual

        cls.I = cls.e(*I_tuple)
        cls.O = cls.e(0).dual()
        cls.BASIS_VECTORS = tuple(cls.e(i) for i in range(cls._dim))
        cls.BASIS_MULTIVECTORS = tuple(cls.e(*b) for b in _basis(cls._dim))
    def __init__(self, scalar: float = 0, *, data: NDArray[np.float64] | None = None) -> None:
        if data is None:
            self._data = np.zeros(type(self)._basis_count)
            self._data[0] = scalar
        else:
            assert scalar == 0
            self._data = data.copy()
    def __add__(self, other: float | Self) -> Self:
        "addition is component-wise"
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = self._data + other._data)
    def __mul__(self, other: float | Self) -> Self:
        """
        The geometric product
        defined by:
        e0 * e0 = 0
        eN * eN = 1 for N != 0
        eN * eM = - eM * eN for N != M
        where N, M: int
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in self._geometric_mul:
            result._data[out] += sign * self._data[in1] * other._data[in2]
        return result
    def __xor__(self, other: float | Self) -> Self:
        """
        The outer product/wedge product
        defined by:
        eN * eN = 0
        eN * eM = - eM * eN for N != M
        where N, M: int

        The most common usecase for ^ is to intersect geometrical elements
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in self._wedge_mul:
            result._data[out] += sign * self._data[in1] * other._data[in2]
        return result
    def __repr__(self) -> str:
        """
        examples:
        1.0e1 + 1.0e23
        2.5 + 1.0e0 + 2.7e1
        """
        r = []
        if self._data[0] != 0:
            r.append(f'{self._data[0]}')
        for n, e in zip(self._data[1:], _basis(self._dim)[1:]):
            if n != 0:
                r.append(str(n) + 'e' + ''.join(str(m) for m in e))
        return ' + '.join(r) if len(r) != 0 else '0'
    def __radd__(self, other: float | Self) -> Self:
        "addition is component-wise"
        return self + other
    def __sub__(self, other: float | Self) -> Self:
        "subtraction is component-wise"
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = self._data - other._data)
    def __rsub__(self, other: float | Self) -> Self:
        "subtraction is component-wise"
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = other._data - self._data)
    def __rmul__(self, other: float | Self) -> Self:
        """
        The geometric product
        defined by:
        e0 * e0 = 0
        eN * eN = 1 for N != 0
        eN * eM = - eM * eN for N != M
        where N, M: int
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return other * self
    def __rxor__(self, other: float | Self) -> Self:
        """
        The outer product/wedge product
        defined by:
        eN * eN = 0
        eN * eM = - eM * eN for N != M
        where N, M: int

        The most common usecase for ^ is to intersect geometrical elements
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return other ^ self
    def reverse(self) -> Self:
        """
        The reverse of a versor
        for a versor
          A = a1 * … * aN
        where a1, …, aN are _vectors_
        is defined as
          A.reverse() = aN * … * a1
        extended linearly to nonversors
        """
        return type(self)(data = self._reverse * self._data)
    def sqr_norm(self) -> float:
        """
        Returns self * self.reverse()
        Only works on versors
        """
        n = self * self.reverse()
        if not np.allclose(n._data[1:], 0):
            raise ValueError('cannot take a norm of a non-versor')
        return abs(n._data[0])
    def norm(self) -> float:
        """
        Returns sqrt(self * self.reverse())
        Only works on versors
        """
        return math.sqrt(self.sqr_norm())
    def normalize(self) -> Self:
        """
        Returns self/self.norm()
        Supports only non-ideal versors
        """
        n = self.norm()
        if n == 0:
            raise ValueError('Cannot normalize an ideal element')
        return self/n 
    def inv(self) -> Self:
        """
        The inverse element of a versor
        defined by
          A * A.inv() == 1 == A.inv() * A
        where A is a versor
        Only supports non-ideal versors
        """
        rev = self.reverse()
        n = self * rev
        if not np.allclose(n._data[1:], 0):
            raise ValueError('cannot take a norm of a non-versor')
        if n._data[0] == 0:
            raise ValueError('cannot take a norm of ideal element')
        return rev * (1/n._data[0])
    def __truediv__(self, other: float | Self) -> Self:
        """
        Right division:
          A / B = A * B.inv()
        where A is a multivector and B is a versor
        Supports only non-ideal versors
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return self * other.inv()
    def __rtruediv__(self, other: float | Self) -> Self:
        """
        Right division:
          A / B = A * B.inv()
        where A is a multivector and B is a versor
        Supports only non-ideal versors
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return other * self.inv()
    def __eq__(self, other: object) -> bool:
        "elementwise equality"
        if isinstance(other, float | int):
            other = type(self)(other)
        if isinstance(other, type(self)):
            return (self._data == other._data).all()
        else:
            return NotImplemented
    def almost_eq(self, other: float | Self) -> bool:
        "== with epsilon for comparing floats"
        if isinstance(other, float | int):
            other = type(self)(other)
        return np.allclose(self._data, other._data)
    def __neg__(self) -> Self:
        "negation is component-wise"
        return type(self)(data = -self._data)
    def dual(self) -> Self:
        """
        the inverse of undual()
        A * A.dual() == α*I
        where
          α = A * A.reverse()
        or
          α = A.dual() * A.dual().reverse()
        depending on which of them isn't zero
        """
        return type(self)(data = (self._dual * self._data)[::-1])
    def undual(self) -> Self:
        """
        the inverse of dual()
        A.undual() * A == α*I
        where
          α = A * A.reverse()
        or
          α = A.undual() * A.undual().reverse()
        depending on which one isn't zero
        """
        return type(self)(data = (self._undual * self._data)[::-1])
    def __and__(self, other: float | Self) -> Self:
        """
        The join operator
        defined as
          A & B == (A.dual() ^ B.dual()).undual()
        it can join
          2 points into a line
          3 points into a plane
          a point and a line into a plane
        it can also be used for calculating volume, area, etc.  
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return (self.dual() ^ other.dual()).undual()
    def __or__(self, other: float | Self) -> Self:
        """
        The left contraction operator
          a & b = a * b
          a & (b ^ C) = a & b - a & C
          (a ^ b) & C = a & (b & C)
        where a, b are vectors and C is a blade
        It's linearly extended to non-blade operands
        Note: it's not associative

        If you interpret geometrical elements as subspaces
          plane -> vectors passing through that plane
          line  -> vectors passing through that line
          point -> vectors passing through that point
        then A & B gives you a subspace of B that is orthogonal to A

        It's commonly used for projecting one element onto another one
        The formula for projecting is
          (A & B) / A        for A.grade() < B.grade()
          A.inv() * (B & A)  for A.grade() > B.grade()
        where A and B are blades
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in self._contraction_mul:
            result._data[out] += sign * self._data[in1] * other._data[in2]
        return result
    def grade(self) -> int | None:
        """
        Returns the grade of self or None if self is mixed grade
        when self is 0 returns None

        Grade is defined as
          (α * (a_1 ^ … ^ a_n)).grade() == n
        where α is a scalar and a_1, …, a_n are vectors
        """
        result = None
        for i in range(self._basis_count):
            if abs(self._data[i] - 0) < 1e-10: continue
            tmp = i.bit_count()
            if result is None:
                result = tmp
            elif result != tmp:
                return None
        return result
    @classmethod
    def e(cls, *ns: int) -> Self:
        """
        Constucts the specified basis blade

        e(i_1, …, i_n) == e(i_1) * … * e(i_n)
        e(i) is the ith basis vector

        There's no nice way that would allow you to write
        things like e12, e012, etc. without defining all
        of them.
            e(1, 2) means e12 (which is a shorthand for e1 * e2)
            e(0, 1) means e01 (which is a shorthand for e0 * e1)
            e(0)    means e0
        etc.
        the pga.in2d and pga.in3d modules export e0, e1, …
        but for more complex blades you can use this function (or just multiply them manually)
        """
        r = cls()
        sign, ix = _to_ix(ns, squares = cls._squares)
        r._data[ix] = sign
        return r


class Geo3d(GeoBase, dimension = 3):
    """
    A number from 3D PGA

    You can use it to represent:
    - planes
    - lines
    - points
    - euclidian motions

    Overloaded operators:
    - A ^ B -> meet
    - A & B -> join
    - A | B -> left contraction
    """
    pass
class Geo2d(GeoBase, dimension = 2):
    """
    A number from 2D PGA

    You can use it to represent:
    - lines
    - points
    - euclidian motions

    Overloaded operators:
    - A ^ B -> meet
    - A & B -> join
    - A | B -> left contraction
    """
    pass
"""
A type variable for dimension agnostic functions
"""
Geo = TypeVar('Geo', Geo3d, Geo2d)

# this implementation follows Course notes Geometric Algebra for Computer Graphics SIGGRAPH 2019 Charles G. Gunn, Ph. D.
_I_3d = Geo3d.e(*range(Geo3d._dim))
def _sqrt_dual(D: Geo3d) -> Geo3d:
    sqrt_s = math.sqrt(D._data[0])
    return sqrt_s + _I_3d*D._data[-1]/(2*sqrt_s)
def _inv_dual(D: Geo3d) -> Geo3d:
    m = Geo3d(data = D._data)
    m._data[-1] *= -1
    m._data /= m._data[0]*m._data[0]
    return m
def _split(B: Geo3d) -> tuple[float, float, Geo3d]:
    BB = -B*B
    if BB.almost_eq(0):
        B_ = B.reverse().undual()
        n = B_.norm()
        return 0, n, B_/n
    B_norm = _sqrt_dual(-B*B)
    B_normalized = _inv_dual(B_norm) * B
    return B_norm._data[0], B_norm._data[-1], B_normalized
def _exp_3d(B: Geo3d) -> Geo3d:
    u, v, B_ = _split(B)
    return (math.cos(u) + math.sin(u)*B_)*(1 + v*_I_3d*B_)
def _log_3d(m: Geo3d) -> Geo3d:
    """
    takes a logarithm of a bivector
    _exp_3d(_log_3d(m)) == m
    """
    s1 = m._data[0]
    p1 = m._data[-1]
    m2 = Geo3d(data = m._data)
    m2._data[0] = 0
    m2._data[-1] = 0
    s2, p2, m_ = _split(m2)
    if abs(s1 - 0) < 1e-10:
        u = math.atan2(s2, s1)
        v = p2/s1
    else:
        u = math.atan2(-p1, p2)
        v = -p1/s2
    return (u + v*_I_3d)*m_

def _exp_2d(B : Geo2d) -> Geo2d:
    BB = (B*B)._data[0]
    if abs(BB - 0) < 1e-10:
        return 1 + B
    elif BB < 0:
        B_norm = math.sqrt(-BB)
        return math.cos(B_norm) + B*(math.sin(B_norm)/B_norm)
    else:
        assert False, 'B*B > 0 cannot occur in PGA'
def _log_2d(m : Geo2d) -> Geo2d:
    s = m._data[0]
    B = Geo2d(data = m._data)
    B._data[0] = 0
    BB = (B*B)._data[0]
    if abs(BB - 0) < 1e-10:
        assert s == 1
        return B
    elif BB < 0:
        B_norm = math.sqrt(-BB)
        angle = math.atan2(B_norm, s)
        return B*(angle/B_norm)
    else:
        assert False, "in 2d PGA bivectors can't square to a number greater than one"
    
def exp(m: Geo) -> Geo:
    """
    The exponential function generalized to bivectors
    exp(n*A) == exp(A) * … * exp(A)
                '--------.-------'
                      n times
    Supports only bivectors
    exp(A) where A is a 2-blade represents 
    - rotation if A is not ideal
    - translation if A is ideal
    such that exp(A) leaves A invariant
    """
    if m.grade() != 2:
        raise ValueError('exp only supports bivectors')
    if isinstance(m, Geo2d):
        return _exp_2d(m) # type: ignore # for some reason mypy thinks Geo = Geo3d, I'm not sure why. The asymmetric behaviour makes me think that it might be a bug in mypy but maybe I'm missing something
    elif isinstance(m, Geo3d):
        return _exp_3d(m)
    else:
        raise ValueError('only Geo2d and Geo3d are supported')
def log(m: Geo) -> Geo:
    """
    Takes the logarithm of a versor
    exp(log(m)) == m
    Gives nonsensical results for nonversors
    Returns a bivector (that doesn't have to be a 2-blade)
    """
    if isinstance(m, Geo2d):
        return _log_2d(m) # type: ignore # same reason as in exp()
    elif isinstance(m, Geo3d):
        return _log_3d(m)
    else:
        raise ValueError('only Geo2d and Geo3d are supported')


@overload
def _coords(P: Geo2d) -> tuple[float, float]:
    ...
@overload
def _coords(P: Geo3d) -> tuple[float, float, float]:
    ...
def _coords(P: Geo2d | Geo3d) -> tuple[float, float] | tuple[float, float, float]:
    p = P.undual()
    s = p._data[1 << 0]
    if s == 0:
        raise ValueError(f'cannot draw an ideal point {P}')
    if isinstance(P, Geo2d):
        return p._data[1 << 1]/s, p._data[1 << 2]/s
    return p._data[1 << 1]/s, p._data[1 << 2]/s, p._data[1 << 3]/s

def _2_points_on_a_line(l: Geo) -> tuple[Geo, Geo]:
    l = l.normalize()
    mid = l.inv()*(l | l.O)
    dist = 3
    M = exp(dist*l*l.I)
    a = M * mid / M
    M = exp(-dist*l*l.I)
    b = M * mid / M
    return a, b
_PLANE_RANGE : Final = 3
def _4_points_on_a_plane(m: Geo3d) -> tuple[Geo3d, Geo3d, Geo3d, Geo3d]:
    e0, e1, e2, e3 = Geo3d.BASIS_VECTORS
    l1, l2, l3 = (abs((m | e)._data[0]) for e in (e1, e2, e3))
    d = max(l1, l2, l3)
    if d == l2:
        a, b = e1, e3
    elif d == l1:
        a, b = e2, e3
    else:
        a, b = e1, e2
    min1 = a - _PLANE_RANGE * e0
    max1 = a + _PLANE_RANGE * e0
    min2 = b - _PLANE_RANGE * e0
    max2 = b + _PLANE_RANGE * e0
    return (min1 ^ min2 ^ m,
            min1 ^ max2 ^ m,
            max1 ^ max2 ^ m,
            max1 ^ min2 ^ m)

class Figure2d:
    """
    Collects points, lines, etc. and plots them with matplotlib
    """
    _points : list[tuple[float, float, str | None]]
    _lines : list[tuple[tuple[float, float], tuple[float, float], str | None]]
    _polys : list[list[tuple[float, float]]]
    def __init__(self) -> None:
        self._points = []
        self._lines = []
        self._polys = []
    def draw(self, m: Geo2d | list[Geo2d], label: str | None = None) -> None:
        """
        Supports:
        - lines (vectors)
        - points (2-blades)
        - polygons (list of points)
        """
        if isinstance(m, list):
            points = [_coords(p) for p in m]
            self._polys.append(points)
            return
        dim = 3
        g = m.grade()
        if g == dim-1:
            x, y = _coords(m)
            self._points.append((x, y, label))
        elif g == dim-2:
            A, B = _2_points_on_a_line(m)
            x1, y1 = _coords(A)
            x2, y2 = _coords(B)
            self._lines.append(((x1, x2), (y1, y2), label))
        else:
            raise ValueError(f"error: don't know how to draw {m}")
    def show(self) -> None:
        """
        Uses matplotlib to show the fugure
        """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.axis('equal')
        for points2 in self._polys:
            xs1 = []
            ys1 = []
            for x, y in points2:
                xs1.append(x)
                ys1.append(y)
            ax.fill(xs1, ys1, alpha=.5)
        for xs, ys, label in self._lines:
            ax.plot(xs, ys)
            if label is not None:
                x = sum(xs)/2
                y = sum(ys)/2
                ax.annotate(label,
                    xy=(x, y), xycoords='data',
                    xytext=(4, 4), textcoords='offset points')
        for x, y, label in self._points:
            ax.plot(x, y, 'ro')
            if label is not None:
                # plt.annotate(label, xy=(x, y))
                ax.annotate(label,
                        xy=(x, y), xycoords='data',
                        xytext=(4, 4), textcoords='offset points')
        fig.show()


_COLORS = tuple(colors.BASE_COLORS.values()) 
class Figure3d:
    """
    Collects points, lines, etc. and plots them with matplotlib
    """
    _points : list[tuple[float, float, float, str | None]] = []
    _lines : list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    _polys : list[list[tuple[float, float, float]]] = []
    def __init__(self) -> None:
        self._points = []
        self._lines = []
        self._polys = []
    def draw(self, m: list[Geo3d] | Geo3d, label: str | None = None) -> None:
        """
        Supports:
        - planes (vectors)
        - lines (2-blades)
        - points (3-blades)
        - polygons (list of points)
        """
        if isinstance(m, list):
            points = [_coords(A) for A in m]
            self._polys.append(points)
            return
        g = m.grade()
        if g == 3:
            xyz = _coords(m)
            self._points.append((*xyz, label))
        elif g == 2:
            A, B = _2_points_on_a_line(m)
            x1, y1, z1 = _coords(A)
            x2, y2, z2 = _coords(B)
            self._lines.append(((x1, x2), (y1, y2), (z1, z2)))
        elif g == 1:
            points = [_coords(A) for A in  _4_points_on_a_plane(m)]
            self._polys.append(points)
        else:
            print(f"don't know how to draw {m}")

    def show(self) -> None:
        """
        Uses matplotlib to show the fugure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-_PLANE_RANGE, _PLANE_RANGE)
        ax.set_ylim(-_PLANE_RANGE, _PLANE_RANGE)
        ax.set_zlim(-_PLANE_RANGE, _PLANE_RANGE) # type: ignore # there's probably a missing annotation in matplotlib
        x = [p[0] for p in self._points]
        y = [p[1] for p in self._points]
        z = [p[2] for p in self._points]
        ax.scatter(x, y, z)
        for l in self._lines:
            ax.plot(*l)
        for i, points in enumerate(self._polys):
            points2 = [[list(elem) for elem in points]]
            tmp = Poly3DCollection(points2, alpha=.5)
            tmp.set_color(_COLORS[i % len(_COLORS)])
            ax.add_collection3d(tmp) # type: ignore # there's probably a missing annotation in matplotlib
        fig.show()

_figure : None | Figure2d | Figure3d = None
def draw(m: list[Geo], label: str | None = None) -> None:
    """
    Draws a geometric element onto a (global) figure
    to show the figure, call show()
    
    Infers if the figure should be 2D or 3D based on
    the elements passed to it (mixing 2D and 3D is not allowed)

    2D supports:
    - lines (vectors)
    - points (2-blades)
    - polygons (lists of points)

    3D supports:
    - planes (vectors)
    - lines  (2-blades)
    - points (3-blades)
    - polygons (lists of points)
    """
    global _figure
    if (islist := isinstance(m, list)) and not len(m) > 1:
        raise ValueError('cannot draw an empty polygon')
    match m[0] if islist else m, _figure:
        case Geo2d(), Figure2d() | None:
            _figure = _figure or Figure2d()
            _figure.draw(m, label) # type: ignore # Am I doing something wrong or is mypy this bad at working with type variables constrained to a particular type? i'd expect the match to generate a Geo = Geo2d constraint, but it seem that mypy only supports that kind of thing for variables
        case Geo3d(), Figure3d() | None:
            _figure = _figure or Figure3d()
            _figure.draw(m, label) # type: ignore # if I used overloads instead of type vars, mypy could probably handle this — which raises the question: why does it treat overloads and type vars with constraints on them differently?
        case _:
            raise ValueError('cannot mix 2D and 3D')
def show() -> None:
    """
    Uses matplotlib to show the (global) figure that draw() draws on
    Then it clears the (global) figure
    """
    global _figure
    if _figure is None:
        raise ValueError('cannot show anything because draw() was not called')
    _figure.show()
    _figure = None
# - references
