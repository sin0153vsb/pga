# An implementation of Projective Geometric Algebra

This library implements operations from 2D and 3D PGA and supports visualization using matplotlib.

Supported operations:

```
e(1, 2)    # basis blade e12
A * B      # geometric product
A ^ B      # meet (intersection)
A & B      # join
A | B      # left contraction
A.dual()   # hedge duality
A.undual() # inverse of dual()
A.inv()    # inverse of a versor
A.reverse()
A.normalized()
exp(A)     # exponential of a bivector
log(A)     # logarithm of a versor
```

## What's PGA?
PGA is to Euclidian motion what complex numbers are to 2D rotations. 

It can uniformly express planes, lines, points, rotations, translations and reflections in a dimension agnostic manner.
For instance to reflect around a plane `a` you use this equation:
```
X' = a*X/a
```
where `X` can be a plane, line, point, or generally any n-dimendional hyperplane, or even a rotation or reflection.
You can combine many reflections into a single motor that can represent arbitrary rigid body motions

For more information see the tutorial in the examples folder

## Hello world

```
from pga.in3d import * # pick dimension
plane1 = e1 + e2 - e3
plane2 = e2 - 2*e0
draw(plane1)
draw(plane2)
draw(plane1 ^ plane2) # ^ is intersection
show()
```

## This implementation is based on

Leo Dorst & Steven De Keninck  
A Guided Tour to the Plane-Based Geometric Algebra PGA  
2022, version 2.0  
Available at http://www.geometricalgebra.net  
and http://bivector.net/PGA4CS.html.

Course notes Geometric Algebra for Computer Graphics  
SIGGRAPH 2019  
Charles G. Gunn, Ph. D.

Geometric Algebra for Computer Science  
An Object Oriented Approach to Geometry  
Leo Dorst, Daniel Fontijne, Stephen Mann

