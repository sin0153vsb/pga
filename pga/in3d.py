"""
This module reexports things for working in 3D
"""
from ._geo import Geo3d, exp, log, draw, show

e = Geo3d.e
e0 = e(0)
e1 = e(1)
e2 = e(2)
e3 = e(3)
I = Geo3d.I
O = Geo3d.O

__all__ = ['O', 'I', 'e0', 'e1', 'e2', 'e3', 'e', 'Geo3d', 'exp', 'log', 'draw', 'show']
