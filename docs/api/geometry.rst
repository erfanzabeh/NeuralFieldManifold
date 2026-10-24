Geometric Fitting
=================

Fit elliptical rings or tori to three-dimensional clouds and compare
trajectories with Procrustes disparity. Fits retain the input coordinate
scale. Containment score and outside distance quantify geometric fit;
they do not establish topological recovery.

Ring and Torus Fits
-------------------

.. autofunction:: neuralfieldmanifold.geometry.fit_ring

.. autofunction:: neuralfieldmanifold.geometry.fit_torus

Settings and Results
--------------------

.. autoclass:: neuralfieldmanifold.geometry.FitConfig
   :members:

.. autoclass:: neuralfieldmanifold.geometry.GeometryFit
   :members:

Trajectory Comparison
---------------------

.. autofunction:: neuralfieldmanifold.geometry.pairwise_disparity

.. autoclass:: neuralfieldmanifold.geometry.Disparity
   :members:
