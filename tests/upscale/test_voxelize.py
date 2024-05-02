"""
Test of homogenization algorithms from voxelize.py
- Homogenization of bulk constant conductivity + discreate fractures with size dependent conductivity.
  Reference is decovalex slow solution modified for anisotropic regular grid.
  This assigns The same conductivity to all intersection cells.

  In order to develop more precises homogenization techniques, we must use two-scale test problems.
"""
from typing import *
import yaml
import shutil
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


import numpy as np
import attrs
import pyvista as pv

from bgem import stochastic
from bgem import fn
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import Grid, Fe, voigt_to_tn, tn_to_voigt, FracturedMedia, voxelize
from bgem.voxelize import EllipseShape, PolygonShape

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"
from joblib import Memory
memory = Memory(workdir, verbose=0)



def test_EllipseShape():
    ellipse = EllipseShape(6)  # Create a hexagon for testing

    # Points to test
    inside_point = (0.5, 0.5)
    edge_point = (np.cos(np.pi / 6), np.sin(np.pi / 6))  # A point exactly on the edge
    outside_point = (1.5, 1.5)

    # Test for a point inside
    assert ellipse.is_point_inside(*inside_point), "The point should be inside the ellipse"

    # Test for a point on the edge
    # Depending on your implementation's edge handling, this could be either True or False
    # Here we assume points on edges are considered inside
    assert ellipse.is_point_inside(*edge_point), "The point should be considered inside the ellipse"

    # Test for a point outside
    assert not ellipse.is_point_inside(*outside_point), "The point should be outside the ellipse"

    # Test points: inside, on edge, outside
    points = np.array([
        [0.5, 0.5],  # Inside
        [-0.5, -0.5],  # Inside
        [0.86, 0.5],  # On edge (approximately, considering rounding)
        [1, 1],  # Outside
        [0, 0]  # Centrally inside
    ])

    expected_results = np.array([
        True,  # Inside
        True,  # Inside
        False,  # On edge, but due to precision, might be considered outside
        False,  # Outside
        True  # Centrally inside
    ])

    actual_results = ellipse.are_points_inside(points)
    np.testing.assert_array_equal(actual_results, expected_results,
                                  "The are_points_inside method failed to accurately determine if points are inside the ellipse.")


def test_PolyShape():
    polygon = PolygonShape(6)  # Create a hexagon for testing

    # Points to test
    inside_point = (0.5, 0.5)
    edge_point = (np.cos(np.pi / 6), np.sin(np.pi / 6))  # A point exactly on the edge
    outside_point = (1.5, 1.5)

    # Test for a point inside
    assert polygon.is_point_inside(*inside_point), "The point should be inside the polygon"

    # Test for a point on the edge
    # Depending on your implementation's edge handling, this could be either True or False
    # Here we assume points on edges are considered inside
    assert polygon.is_point_inside(*edge_point), "The point should be considered inside the polygon"

    # Test for a point outside
    assert not polygon.is_point_inside(*outside_point), "The point should be outside the polygon"

    # Test points: inside, on edge, outside
    points = np.array([
        [0.5, 0.5],  # Inside
        [-0.5, -0.5],  # Inside
        [0.86, 0.5],  # On edge (approximately, considering rounding)
        [1, 1],  # Outside
        [0, 0]  # Centrally inside
    ])

    expected_results = np.array([
        True,  # Inside
        True,  # Inside
        False,  # On edge, but due to precision, might be considered outside
        False,  # Outside
        True  # Centrally inside
    ])

    actual_results = polygon.are_points_inside(points)
    np.testing.assert_array_equal(actual_results, expected_results,
                                  "The are_points_inside method failed to accurately determine if points are inside the polygon.")


def dfn_4_fractures():
    return voxelize.FracturedMedia.from_dfn_works(script_dir / "4_fractures", 0.01)

def tst_fracture_set(domain):
    R = 2*np.max(domain)
    fr = lambda c, n : stochastic.Fracture(stochastic.SquareShape, R, c, n, 0.0, 123, 1)
    return [
        #fr([0, 0, 0.7], [0, 0, 1]),
        #fr([0, 0.7, 0], [0, 1, 0]),
        #fr([0.7, 0, 0], [1, 0, 0]),
        #fr([0, 0, 0], [0.5, 0, 1]),
        fr([0, 0, 0.7], [0, 0.5, 1]),
        #fr([0, 0, 0], [0.1, 1, 1]),
        #fr([0, 0, 0], [0.3, 1, 1]),
        #fr([0, 0, -0.7], [0.5, 1, 1]),
        fr([0, 0, -0.5], [1, 1, 1])
           ]



def homo_decovalex(fr_media: FracturedMedia, grid:Grid, perm_fn):
    """
    Homogenize fr_media to the conductivity tensor field on grid.
    :return: conductivity_field, np.array, shape (n_elements, n_voight)
    """
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    fractures = dmap.map_dfn(d_grid, ellipses)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    return perm_fn(d_grid, fractures, fr_transmissivity, fr_media.conductivity)

def homo_decovalex_iso(fr_media: FracturedMedia, grid:Grid):
    perm_fn = lambda *args : dmap.permIso(*args)[:, None, None] * np.eye(3)
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_raw(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.permAnisoRaw(*args)
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_diag(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.aniso_diag(dmap.permAnisoRaw(*args))
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_lump(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.aniso_lump(dmap.permAnisoRaw(*args))
    return homo_decovalex(fr_media, grid, perm_fn)

def rasterize_dfn(homo_fns):
    # Fracture set
    domain_size = 100

    # Coarse Problem
    steps = (10, 12, 14)
    grid = Grid(domain_size, steps, Fe.Q(dim=3), origin=-domain_size / 2)
    dfn = tst_fracture_set(grid.dimensions)
    fr_media = FracturedMedia.fracture_cond_params(dfn, 0.1, 1)

    xyz_range = [ np.linspace(grid.origin[ax], grid.origin[ax] + grid.dimensions[ax], grid.shape[ax] + 1, dtype=np.float32)
                  for ax in [0, 1, 2]
                ]

    x, y, z = np.meshgrid(*xyz_range, indexing='ij')
    pv_grid = pv.StructuredGrid(x, y, z)
    #points = grid.nodes()
    for name, homo_fn in homo_fns.items():
        grid_permitivity = homo_fn(fr_media, grid)
        if len(grid_permitivity.shape) > 1:
            # anisotropic case
            assert grid_permitivity.shape[1:] == (3, 3)
            grid_permitivity = grid_permitivity.reshape(-1, 9)
        pv_grid.cell_data[name] = grid_permitivity
    pv_grid.save(str(workdir / "test_resterize.vtk"))


def test_load_dfnworks():
    dfn = dfn_4_fractures()
    assert dfn.dfn.size == 4

def test_reasterize():
    homo_fns=dict(
        k_deco_iso=voxelize.permeability_iso_decovalex,
        k_deco_aniso_raw=voxelize.permeability_aniso_decovalex,
        k_deco_aniso_diag=fn.compose(voxelize.aniso_diag, voxelize.permeability_aniso_decovalex),
        k_deco_aniso_lump=fn.compose(voxelize.aniso_lump, voxelize.permeability_aniso_decovalex)
    )
    rasterize_dfn(homo_fns)
