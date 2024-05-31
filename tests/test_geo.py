from pga import *
import pytest

def test_log_of_exp_2d() -> None:
    basis_bivectors_2d = (Geo2d.e(0,1), Geo2d.e(0,2), Geo2d.e(1, 2))
    for a in basis_bivectors_2d:
        for b in basis_bivectors_2d:
            B = a + b
            assert log(exp(B)).almost_eq(B)
def test_log_of_exp_3d() -> None:
    e = Geo3d.e
    basis_bivectors_3d = (e(0,1), e(0,2), e(0,3), e(1,2), e(1,3), e(2,3))
    for a in basis_bivectors_3d:
        for b in basis_bivectors_3d:
            B = a + .2*b
            assert log(exp(B)).almost_eq(B)

def test_assoc_3d() -> None:
    bs = Geo3d.BASIS_MULTIVECTORS
    for a in bs:
        for b in bs:
            for c in bs:
                assert (a * b) * c == a * (b * c)
                assert (a ^ b) ^ c == a ^ (b ^ c)
                assert (a & b) & c == a & (b & c)
def test_distributivity_3d() -> None:
    bs = Geo3d.BASIS_MULTIVECTORS
    for a in bs:
        for b in bs:
            for c in bs:
                assert (a + b) * c == a * c + b * c
                assert (a + b) ^ c == (a ^ c) + (b ^ c)
                assert (a + b) & c == (a & c) + (b & c)
def test_antisymmetry_3d() -> None:
    bs = Geo3d.BASIS_VECTORS
    for a in bs:
        for b in bs:
            assert (a ^ b) == -(b ^ a)
            assert (a & b) == -(b & a)
def test_split_geometric_mult_to_wedge_and_contraction_3d() -> None:
    bs = Geo3d.BASIS_VECTORS
    for a in bs:
        for b in bs:
            assert (a * b) == (a | b) + (a ^ b)
def test_dual_is_an_isomorphism_3d() -> None:
    bs = Geo3d.BASIS_MULTIVECTORS
    for a in bs:
        assert a * a.dual() == Geo3d.I
        assert a == a.dual().undual()


def test_assoc_2d() -> None:
    bs = Geo2d.BASIS_MULTIVECTORS
    for a in bs:
        for b in bs:
            for c in bs:
                assert (a * b) * c == a * (b * c)
                assert (a ^ b) ^ c == a ^ (b ^ c)
                assert (a & b) & c == a & (b & c)
def test_distributivity_2d() -> None:
    bs = Geo2d.BASIS_MULTIVECTORS
    for a in bs:
        for b in bs:
            for c in bs:
                assert (a + b) * c == a * c + b * c
                assert (a + b) ^ c == (a ^ c) + (b ^ c)
                assert (a + b) & c == (a & c) + (b & c)
def test_antisymmetry_2d() -> None:
    bs = Geo2d.BASIS_VECTORS
    for a in bs:
        for b in bs:
            assert (a ^ b) == -(b ^ a)
            assert (a & b) == -(b & a)
def test_split_geometric_mult_to_wedge_and_contraction_2d() -> None:
    bs = Geo2d.BASIS_VECTORS
    for a in bs:
        for b in bs:
            assert (a * b) == (a | b) + (a ^ b)
def test_dual_is_an_isomorphism_2d() -> None:
    bs = Geo2d.BASIS_MULTIVECTORS
    for a in bs:
        assert a * a.dual() == Geo2d.I
        assert a == a.dual().undual()
