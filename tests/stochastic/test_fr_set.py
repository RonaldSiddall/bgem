import numpy as np
import pytest
from bgem import stochastic
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


"""
Test base shapes.
"""
def test_ellipse_shape():
    shape = stochastic.EllipseShape



def plot_aabb(aabb, points, inside):
    # Prepare the AABB rectangle coordinates
    aabb_rect = np.array([
        aabb[ii, (0,1)] for ii in [(0,0), (1, 0), (1, 1), (0, 1), (0,0)]
    ])

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(inside, 'red', 'grey')
    ax.scatter(points[:, 0], points[:, 1],s=1, color=colors, label='Points')
    ax.plot(aabb_rect[:, 0], aabb_rect[:, 1], color='blue', linestyle='--', label='AABB')
    ax.set_aspect('equal', 'box')
    # Labels and Title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot with AABB')
    ax.legend()

    # Display the plot
    ax.grid(True)
    plt.show()

@pytest.mark.parametrize("base_shape",
    [stochastic.EllipseShape(), stochastic.RectangleShape(), stochastic.PolygonShape(6), stochastic.PolygonShape(8)]
)
def test_base_shapes(base_shape):
    """
    Use MC integration to:
    - confirm the shape has unit area
    - check it could determine interrior points (but not that this check is correct)
    - check that the corresponding primitive could be made in GMSH interface
    - confirm that aabb is correct for that primitive
    :param shape:
    :return:
    """

    # Take AABB of the reference shape
    aabb = base_shape.aabb
    assert aabb.shape == (2, 2)
    assert np.allclose(aabb[0] + aabb[1], 0)
    safe_aabb = 2 * aabb

    # equivalence of is_point_inside ande are_points_inside
    N = 1000
    points = np.random.random((N, 2)) * (safe_aabb[1] - safe_aabb[0]) + safe_aabb[0]
    inside_single = [base_shape.is_point_inside(*pt) for pt in points]
    inside_vector = base_shape.are_points_inside(points)

    assert np.all(np.array(inside_single, dtype=bool) == inside_vector)
    out_of_aabb = np.logical_or.reduce((*(points < aabb[0]).T,  *(aabb[1] < points).T))
    any_out = np.any(inside_vector & out_of_aabb)
    #if any_out:
    #plot_aabb(aabb, points, inside_vector)
    assert not any_out


    N = 100000
    points = np.random.random((N, 2)) * (aabb[1] - aabb[0]) + aabb[0]
    N_in = sum(base_shape.are_points_inside(points))
    aabb_area = np.prod(aabb[1] - aabb[0])
    area_estimate = N_in / N * aabb_area
    assert abs(area_estimate - 1.0) < 0.01

def single_frac_transform_mat(fr: stochastic.Fracture):
    dfn = stochastic.FractureSet.from_list([fr])
    base_vectors = dfn.transform_mat @ np.eye(3)
    assert base_vectors.shape == (1, 3, 3)
    ref_base = fr.transform(np.eye(3))
    assert np.allclose(dfn.center[0], fr.center)
    ref_base -= fr.center
    assert np.allclose(base_vectors[0], ref_base)

def test_transform_mat():
    """
    Apply transfrom for
    :return:
    """
    shape_id = stochastic.EllipseShape.id
    fr = lambda s, c, n: stochastic.Fracture(shape_id, np.array(s), np.array(c), np.array(n))
    fractures = [
        fr([2, 3], [1, 2, 3], [0, 0, 1]),
        fr([2, 3], [1, 2, 3], [0, 0, -1]),
        fr([2, 3], [0, 0, 0], [0, 1, 0]),
        fr([2, 3], [0, 0, 0], [0, -1, 0]),
        fr([2, 3], [0, 0, 0], [1, 0, 0]),
        fr([2, 3], [0, 0, 0], [-1, 0, 0]),
        fr([2, 3], [0, 0, 0], [1, 2, 3] / np.linalg.norm([1,2,3])),
        #stochastic.Fracture(shape_id, np.array(s), np.array(c), np.array(n))
    ]
    for fr in fractures:
        single_frac_transform_mat(fr)
