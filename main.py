import math
import numpy as np
from typing import Literal, cast, Union, Generic, TypeVar, Self
from numpy.typing import NDArray




Nat = int # should be >= 0
Index = Nat # index into multivector's data

# basis multivector represented as a tuple
TBasis = tuple[Nat, ...]

Sign = Literal[-1, 0, 1]
Squares = tuple[Sign, ...]

def basis(n: Nat) -> tuple[TBasis, ...]:
    r : tuple[TBasis, ...] = ((),)
    for e in range(n):
        r += tuple(tail + (e,) for tail in r)
    return r

def simplify_tbasis(ns: TBasis, *, squares: Squares) -> tuple[Sign, TBasis]:
    """
    uses the laws
    eN * eN = squares[N]
    eN * eM = - eM * eN  where N != M
    where N, M : Nat

    examples:
    e1 * e2 * e1 ==> - squares[1] * e2
    e2 * e1 ==> - e1 * e2
    e1 * e2 * e0 * e1 ==> - squares[1] * e0 * e1
    """
    sign: Sign = 1
    r: list[Nat] = []
    for e in ns[::-1]:
        for i in range(len(r)):
            if r[i] == e:
                sign = cast(Sign, sign * squares[e])
                r.remove(e)
                break
            elif r[i] > e: 
                r.insert(i, e)
                break
            else:
                sign = cast(Sign, -sign)
        else:
            r.append(e)
    return sign, tuple(r)


def to_ix(ns: TBasis, *, squares: Squares) -> tuple[Sign, Index]:
    sign, ns = simplify_tbasis(ns, squares = squares)
    return sign, sum(1 << n for n in ns)


def mult(a: TBasis, b: TBasis, squares: Squares) -> tuple[Sign, TBasis]:
    return simplify_tbasis(a + b, squares = squares)

def mul_table(squares: tuple[Sign, ...]) -> list[tuple[Index, Literal[1, -1], Index, Index]]:
    result = []
    for a in basis(len(squares)):
        _, a_ix = to_ix(a, squares = squares)
        for b in basis(len(squares)):
            _, b_ix = to_ix(b, squares = squares)
            sign, ab_ix = to_ix(a + b, squares = squares)
            if sign != 0:
                result.append((ab_ix, sign, a_ix, b_ix))
    return result
def contraction_mult(a: TBasis, b: TBasis, squares: tuple[Sign, ...]) -> tuple[Sign, TBasis]:
    sign: Sign = 1
    r = list(b)
    for e in a[::-1]:
        for i in range(len(r)):
            if r[i] == e:
                sign = cast(Sign, sign * squares[e])
                r.remove(e)
                break
            elif r[i] > e:
                sign = 0
                break
            else:
                sign = cast(Sign, -sign)
        else:
            sign = 0
    return sign, tuple(r)
def contraction_table(squares: tuple[Sign, ...]) -> list[tuple[Index, Literal[1, -1], Index, Index]]:
    result = []
    for a in basis(len(squares)):
        _, a_ix = to_ix(a, squares = squares)
        for b in basis(len(squares)):
            _, b_ix = to_ix(b, squares = squares)
            sign, r = contraction_mult(a, b, squares = squares)
            _, r2 = to_ix(r, squares = squares)
            if sign != 0:
                result.append((r2, sign, a_ix, b_ix))
    return result
s0: Sign = 0
s1: Sign = 1

dim = 4
basis_count = 2**dim
squares = (s0,) + (s1,)*(dim - 1)
geometric_mul = mul_table(squares)
wedge_mul = mul_table(squares = (s0,)*dim)
contraction_mul = contraction_table(squares)
reverse : NDArray[np.float64]
dual : NDArray[np.float64]
undual : NDArray[np.float64]
def init() -> None:
    global reverse
    reverse = np.ones(basis_count)
    for i in range(basis_count):
        #       + + - - + + - - ...
        # grade 0 1 2 3 ...
        if i.bit_count() % 4 >= 2:
            reverse[i] = -1

    I_tuple = tuple(range(dim))
    
    global dual
    dual = np.ones(basis_count)
    for i, e in enumerate(basis(dim)):
        sign, d = mult(e[::-1], I_tuple, squares = (s1,)*dim)
        dual[i] = sign

    global undual
    undual = np.ones(basis_count)
    for i, e in enumerate(basis(dim)):
        sign, _ = mult(I_tuple, e[::-1], squares = (s1,)*dim)
        undual[i] = sign
init()

GeoLike = Union['Geo', float]

class Geo:
    """
    a number from PGA
    """
    
    def __init__(self, scalar: float = 0, *, data: NDArray[np.float64] | None = None) -> None:
        if data is None:
            self.data = np.zeros(basis_count)
            self.data[0] = scalar
        else:
            assert scalar == 0
            self.data = data.copy()
    def __add__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = self.data + other.data)
    def __mul__(self, other: float | Self) -> Self:
        """
        The geometric product
        defined by:
        e0 * e0 = 0
        eN * eN = 1 for N != 0
        eN * eM = - eM * eN for N != M
        where N, M: Nat
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in geometric_mul:
            result.data[out] += sign * self.data[in1] * other.data[in2]
        return result
    def __xor__(self, other: float | Self) -> Self:
        """
        The outer product
        defined by:
        eN * eN = 0
        eN * eM = - eM * eN for N != M
        where N, M: Nat
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in wedge_mul:
            result.data[out] += sign * self.data[in1] * other.data[in2]
        return result
    def __repr__(self) -> str:
        r = []
        if self.data[0] != 0:
            r.append(f'{self.data[0]}')
        for n, e in zip(self.data[1:], basis(dim)[1:]):
            if n != 0:
                r.append(str(n) + 'e' + ''.join(str(m) for m in e))
        return ' + '.join(r) if len(r) != 0 else '0'
    def __radd__(self, other: float | Self) -> Self:
        return self + other
    def __sub__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = self.data - other.data)
    def __rsub__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            other = type(self)(other)
        return type(self)(data = other.data - self.data)
    def __rmul__(self, other: float | Self) -> Self:
        """
        The geometric product
        defined by:
        e0 * e0 = 0
        eN * eN = 1 for N != 0
        eN * eM = - eM * eN for N != M
        where N, M: Nat
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return other * self
    def __rxor__(self, other: float | Self) -> Self:
        """
        The outer product
        defined by:
        eN * eN = 0
        eN * eM = - eM * eN for N != M
        where N, M: Nat
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
        return type(self)(data = reverse * self.data)
    def sqr_norm(self) -> float:
        # TODO: doing a full geometric product is an overkill here
        # TODO: error when the result isn't nonzero scalar
        return (self * self.reverse()).data[0]
    def norm(self) -> float:
        # TODO: error when the result isn't nonzero scalar
        # TODO: should the abs() be done by norm() or sqr_norm()?
        return math.sqrt(abs(self.sqr_norm()))
    def normalize(self) -> Self:
        return self/self.norm()
    def inv(self) -> Self:
        """
        The inverse element of a versor
        defined by
          A * A.inv() == 1 == A.inv() * A
        where A is a versor
        for nonversors might give nonsensical results
        don't pass in 0
        """
        # TODO: error when the result isn't nonzero scalar
        rev = self.reverse()
        sqr_norm = self * rev
        return rev * (1/sqr_norm.data[0])
    def __truediv__(self, other: float | Self) -> Self:
        """
        Right division:
          A / B = A * B.inv()
        where A is a multivector and B is a versor
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return self * other.inv()
    def __rtruediv__(self, other: float | Self) -> Self:
        """
        Right division:
          A / B = A * B.inv()
        where A is a multivector and B is a versor
        """
        if isinstance(other, float | int):
            other = type(self)(other)
        return other * self.inv()
    def __eq__(self, other: object) -> bool:
        if isinstance(other, float | int):
            other = type(self)(other)
        if isinstance(other, type(Self)):
            return (self.data == other.data).all()
        else:
            return NotImplemented
    def almost_eq(self, other: float | Self) -> bool:
        if isinstance(other, float | int):
            other = type(self)(other)
        return np.allclose(self.data, other.data)
    def __neg__(self) -> Self:
        return type(self)(data = -self.data)
    def dual(self) -> Self:
        """
        A * A.dual() == α*I
        where
          α = A * A.reverse()
        or
          α = A.dual() * A.dual().reverse()
        depending on which of them isn't 
        """
        return type(self)(data = (dual * self.data)[::-1])
    def undual(self) -> Self:
        return type(self)(data = (undual * self.data)[::-1])
    def __and__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            other = type(self)(other)
        return (self.dual() ^ other.dual()).undual()
    def __or__(self, other: GeoLike) -> Self:
        if isinstance(other, float | int):
            other = type(self)(other)
        result = type(self)()
        for out, sign, in1, in2 in contraction_mul:
            result.data[out] += sign * self.data[in1] * other.data[in2]
        return result
    def sqrt(self) -> Self:
        # TODO: document when this works
        n = self.norm()
        return n*(1 + self/n).normalize()
    def grade(self) -> Nat | None:
        """
        returns the grade of self or None if self is mixed grade
        when self is 0 return None
        """
        result = None
        for i in range(basis_count):
            if self.data[i] == 0: continue
            tmp = i.bit_count()
            if result is None:
                result = tmp
            elif result != tmp:
                return None
        return result
    @classmethod
    def init(cls, *, dimension: Literal[2, 3]) -> None:
        pass

class Geo2d(Geo): pass
Geo2d.init(dimension = 2)
class Geo3d(Geo): pass
Geo2d.init(dimension = 3)

def e(*ns: Nat) -> Geo:
    r = Geo()
    sign, ix = to_ix(ns, squares = squares)
    r.data[ix] = sign
    return r
e0 = e(0)
e1 = e(1)
e2 = e(2)
if dim == 4:
    e3 = e(3)
I = e(*range(dim))

# TODO: remove
# get_grade2 = np.zeros(basis_count)
# for i in range(basis_count):
#     if i.bit_count() == 2:
#         get_grade2[i] = 1
# def log_3d(V):
#     V0 = V.data[0]
#     print(V0)
#     V2 = Geo(data = V.data*get_grade2)
#     print(V2)
#     W = V2/V0
#     W_contr = (W | W.reverse()).data[0]
#     α = math.atan(math.sqrt(W_contr))
#     βI = (W ^ W.reverse()) / (2*math.sqrt(W_contr))
#     L = (W / math.sqrt(W_contr))*(1-(W ^ W.reverse())/(2*W | W.reverse()))
#     return (α + βI)*L

def sqrt_dual(D: Geo) -> Geo:
    sqrt_s = math.sqrt(D.data[0])
    return sqrt_s + I*D.data[-1]/(2*sqrt_s)
def inv_dual(D: Geo) -> Geo:
    m = Geo(data = D.data)
    m.data[-1] *= -1
    m.data /= m.data[0]*m.data[0]
    return m
def split(B: Geo) -> tuple[float, float, Geo]:
    BB = -B*B
    if BB.almost_eq(0):
        B_ = B.reverse().undual()
        n = B_.norm()
        return 0, n, B_/n
    B_norm = sqrt_dual(-B*B)
    B_normalized = inv_dual(B_norm) * B
    return B_norm.data[0], B_norm.data[-1], B_normalized
def exp_3d(B: Geo) -> Geo:
    u, v, B_ = split(B)
    return (math.cos(u) + math.sin(u)*B_)*(1 + v*I*B_)
def log_3d(m: Geo) -> Geo:
    """
    takes a logarithm of a bivector
    exp_3d(log_3d(m)) == m
    """
    s1 = m.data[0]
    p1 = m.data[-1]
    m2 = Geo(data = m.data)
    m2.data[0] = 0
    m2.data[-1] = 0
    s2, p2, m_ = split(m2)
    if s1 != 0: # TODO: float ==
        u = math.atan2(s2, s1)
        v = p2/s1
    else:
        u = math.atan2(-p1, p2)
        v = -p1/s2
    return (u + v*I)*m_

def exp_2d(B):
    BB = (B*B).data[0]
    if BB == 0:
        return 1 + B
    elif BB < 0:
        B_norm = math.sqrt(-BB)
        return math.cos(B_norm) + B*(math.sin(B_norm)/B_norm)
def log_2d(m):
    s = m.data[0]
    B = Geo(data = m.data)
    B.data[0] = 0
    BB = (B*B).data[0]
    if BB == 0:
        assert s == 1
        return B
    elif BB < 0:
        B_norm = math.sqrt(-BB)
        angle = math.atan2(B_norm, s)
        return B*(angle/B_norm)
    else:
        assert False, "in 2d PGA bivectors can't square to a number greater than one"
    
    

# testing
def test() -> None:
    basis_bivectors_2d = (e(0,1), e(0,2))
    for a in basis_bivectors_2d:
        for b in basis_bivectors_2d:
            B = a + b
            assert log_2d(exp_2d(B)).almost_eq(B)
    if dim == 4:
        basis_bivectors_3d = (e(0,1), e(0,2), e(0,3), e(1,2), e(1,3), e(2,3))
        for a in basis_bivectors_3d:
            for b in basis_bivectors_3d:
                B = a + .2*b
                assert log_3d(exp_3d(B)).almost_eq(B)
    basis_vectors = tuple(e(i) for i in range(dim))
    for i, a in enumerate(basis_vectors):
        print('testing', i, '/', dim)
        for b in basis_vectors:
            print(a, b)
            assert (a ^ b) == -(b ^ a)
            assert (a & b) == -(b & a)
            assert (a * b) == (a | b) + (a ^ b)
    basis2 = tuple(e(*b) for b in basis(dim))
    for i, a in enumerate(basis2):
        print('testing', i, '/', 2**dim)
        assert a * a.dual() == I
        assert a == a.dual().undual()
        for b in basis2:
            for c in basis2:
                # print('a = ', a)
                # print('b = ', b)
                # print('c = ', c)
                assert (a * b) * c == a * (b * c)
                assert (a ^ b) ^ c == a ^ (b ^ c)
                assert (a & b) & c == a & (b & c)
                assert (a + b) * c == a * c + b * c
                assert (a + b) ^ c == (a ^ c) + (b ^ c)
                assert (a + b) & c == (a & c) + (b & c)

def coords(P: Geo) -> None | tuple[float, ...]:
    p = P.undual()
    if p.data[1] == 0:
        return None
    return tuple(p.data[1 << i]/p.data[1] for i in range(1, dim))

O = e0.dual()
if dim == 3:
    exp = exp_2d
elif dim == 4:
    exp = exp_3d
else:
    assert False, 'only 2d and 3d are supported'