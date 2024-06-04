"""
This module reexports things for working in 2D
"""
from ._geo import Geo2d, exp, log, draw, show

e = Geo2d.e
e0 = e(0)
e1 = e(1)
e2 = e(2)
I = Geo2d.I
O = Geo2d.O

__all__ = ['O', 'I', 'e0', 'e1', 'e2', 'e',
           'Geo2d', 'exp', 'log', 'draw', 'show']
