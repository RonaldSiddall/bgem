import pytest
import bspline as bs
import numpy as np
import math
import bspline_plot as bs_plot

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class TestSplineBasis:

    def test_find_knot_interval(self):
        """
        test methods:
        - make_equidistant
        - find_knot_interval
        """
        eq_basis = bs.SplineBasis.make_equidistant(2, 100)
        assert eq_basis.find_knot_interval(0.0) == 0
        assert eq_basis.find_knot_interval(0.001) == 0
        assert eq_basis.find_knot_interval(0.01) == 1
        assert eq_basis.find_knot_interval(0.011) == 1
        assert eq_basis.find_knot_interval(0.5001) == 50
        assert eq_basis.find_knot_interval(1.0 - 0.011) == 98
        assert eq_basis.find_knot_interval(1.0 - 0.01) == 99
        assert eq_basis.find_knot_interval(1.0 - 0.001) == 99
        assert eq_basis.find_knot_interval(1.0) == 99

        knots = np.array([0, 0, 0, 0.1880192, 0.24545785, 0.51219762, 0.82239001, 1., 1. , 1.])
        basis = bs.SplineBasis(2, knots)
        for interval in range(2, 7):
            xx = np.linspace(knots[interval], knots[interval+1], 10)
            for j, x in enumerate(xx[:-1]):
                i_found = basis.find_knot_interval(x)
                assert i_found == interval - 2, "i_found: {} i: {} j: {} x: {} ".format(i_found, interval-2, j, x)


    def test_packed_knots(self):
        """
        Test:
         - make_from_packed_knots
         - pack_knots
        :return:
        """
        packed = [(-0.1, 3), (0,1), (1,1), (1.1, 3)]
        basis = bs.SplineBasis.make_from_packed_knots(2, packed)
        assert packed == basis.pack_knots()




    def plot_basis(self, eq_basis):
        n_points = 401
        x_coord = np.linspace(eq_basis.domain[0], eq_basis.domain[1], n_points)

        for i_base in range(eq_basis.size):
            y_coord = [ eq_basis.eval(i_base, x) for x in x_coord ]
            plt.plot(x_coord, y_coord)

        plt.show()


    def test_eval(self):
        #self.plot_basis(bs.SplineBasis.make_equidistant(0, 4))

        knots = np.array([0, 0, 0, 0.1880192, 0.24545785, 0.51219762, 0.82239001, 1., 1. , 1.])
        basis = bs.SplineBasis(2, knots)
        # self.plot_basis(basis)

        eq_basis = bs.SplineBasis.make_equidistant(0, 2)
        assert eq_basis.eval(0, 0.0) == 1.0
        assert eq_basis.eval(1, 0.0) == 0.0
        assert eq_basis.eval(0, 0.5) == 0.0
        assert eq_basis.eval(1, 0.5) == 1.0
        assert eq_basis.eval(1, 1.0) == 1.0

        eq_basis = bs.SplineBasis.make_equidistant(1, 4)
        assert eq_basis.eval(0, 0.0) == 1.0
        assert eq_basis.eval(1, 0.0) == 0.0
        assert eq_basis.eval(2, 0.0) == 0.0
        assert eq_basis.eval(3, 0.0) == 0.0
        assert eq_basis.eval(4, 0.0) == 0.0

        assert eq_basis.eval(0, 0.125) == 0.5
        assert eq_basis.eval(1, 0.125) == 0.5
        assert eq_basis.eval(2, 1.0) == 0.0

        # check summation to one:
        for deg in range(0, 10):
            basis = bs.SplineBasis.make_equidistant(deg, 2)
            for x in np.linspace(basis.domain[0], basis.domain[1], 10):
                s = sum([ basis.eval(i, x) for i in range(basis.size) ])
                assert np.isclose(s, 1.0)

    def fn_supp(self):
        basis = bs.SplineBasis.make_equidistant(2, 4)
        for i in range(basis.size):
            supp = basis.fn_supp(i)
            for x in np.linspace(supp[0] - 0.1, supp[0], 10):
                assert basis.eval(i, x) == 0.0
            for x in np.linspace(supp[0] + 0.001, supp[1] - 0.001, 10):
                assert basis.eval(i, x) > 0.0
            for x in np.linspace(supp[1], supp[1] + 0.1):
                assert basis.eval(i, x) == 0.0

    def test_linear_poles(self):
        eq_basis = bs.SplineBasis.make_equidistant(2, 4)
        poles = eq_basis.make_linear_poles()

        t_vec = np.linspace(0.0, 1.0, 21)
        for t in t_vec:
            b_vals = np.array([ eq_basis.eval(i, t) for i in range(eq_basis.size) ])
            x = np.dot(b_vals, poles)
            assert np.abs( x - t ) < 1e-15

    def check_eval_vec(self, basis, i, t):
        vec = basis.eval_vector(i, t)
        for j in range(basis.degree + 1):
            assert vec[j] == basis.eval(i + j, t)

    def plot_basis_vec(self, basis):
        n_points = 401
        x_coord = np.linspace(basis.domain[0], basis.domain[1], n_points)

        y_coords = np.zeros( (basis.size, x_coord.shape[0]) )
        for i, x in enumerate(x_coord):
            idx = basis.find_knot_interval(x)
            y_coords[idx : idx + basis.degree + 1, i] = basis.eval_vector(idx, x)

        for i_base in range(basis.size):
            plt.plot(x_coord, y_coords[i_base, :])

        plt.show()


    def test_eval_vec(self):
        basis = bs.SplineBasis.make_equidistant(2, 4)
        # self.plot_basis_vec(basis)
        self.check_eval_vec(basis, 0, 0.1)
        self.check_eval_vec(basis, 1, 0.3)
        self.check_eval_vec(basis, 2, 0.6)
        self.check_eval_vec(basis, 3, 0.8)
        self.check_eval_vec(basis, 3, 1.0)

        basis = bs.SplineBasis.make_equidistant(3, 4)
        # self.plot_basis_vec(basis)
        self.check_eval_vec(basis, 0, 0.1)
        self.check_eval_vec(basis, 1, 0.3)
        self.check_eval_vec(basis, 2, 0.6)
        self.check_eval_vec(basis, 3, 0.8)
        self.check_eval_vec(basis, 3, 1.0)



    def check_diff_vec(self, basis, i, t):
        vec = basis.eval_diff_vector(i, t)
        for j in range(basis.degree + 1):
            assert np.abs(vec[j] - basis.eval_diff(i + j, t)) < 1e-15

    def plot_basis_diff(self, basis):
        n_points = 401
        x_coord = np.linspace(basis.domain[0], basis.domain[1], n_points)

        y_coords = np.zeros( (basis.size, x_coord.shape[0]) )
        for i, x in enumerate(x_coord):
            idx = basis.find_knot_interval(x)
            y_coords[idx : idx + basis.degree + 1, i] = basis.eval_diff_vector(idx, x)

        for i_base in range(basis.size):
            plt.plot(x_coord, y_coords[i_base, :])

        plt.show()


    def test_eval_diff_base_vec(self):
        basis = bs.SplineBasis.make_equidistant(2, 4)
        # self.plot_basis_diff(basis)
        self.check_diff_vec(basis, 0, 0.1)
        self.check_diff_vec(basis, 1, 0.3)
        self.check_diff_vec(basis, 2, 0.6)
        self.check_diff_vec(basis, 3, 0.8)
        self.check_diff_vec(basis, 3, 1.0)

        basis = bs.SplineBasis.make_equidistant(3, 4)
        # self.plot_basis_diff(basis)
        self.check_diff_vec(basis, 0, 0.1)
        self.check_diff_vec(basis, 1, 0.3)
        self.check_diff_vec(basis, 2, 0.6)
        self.check_diff_vec(basis, 3, 0.8)
        self.check_diff_vec(basis, 3, 1.0)



class TestCurve:

    def plot_4p(self):
        degree = 2
        poles = [ [0., 0.], [1.0, 0.5], [2., -2.], [3., 1.] ]
        basis = bs.SplineBasis.make_equidistant(degree, 2)
        curve = bs.Curve(basis, poles)

        bs_plot.plot_curve_2d(curve, poles=True)
        b00, b11 = curve.aabb()
        b01 = [b00[0], b11[1]]
        b10 = [b11[0], b00[1]]
        bb = np.array([b00, b10, b11, b01, b00])

        plt.plot( bb[:, 0], bb[:, 1], color='green')
        plt.show()

    def test_evaluate(self):
        # self.plot_4p()
        # TODO: make numerical tests with explicitely computed values
        # TODO: test rational curves, e.g. circle

        pass











class TestSurface:

    def plot_extrude(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # curve extruded to surface
        poles_yz = [[0., 0.], [1.0, 0.5], [2., -2.], [3., 1.]]
        poles_x = [0, 1, 2]
        poles = [ [ [x] + yz for yz in poles_yz ] for x in poles_x ]
        u_basis = bs.SplineBasis.make_equidistant(2, 1)
        v_basis = bs.SplineBasis.make_equidistant(2, 2)
        surface_extrude = bs.Surface( (u_basis, v_basis), poles)
        bs_plot.plot_surface_3d(surface_extrude, ax, poles = True)
        plt.show()

    def plot_function(self):
        # function surface
        def function(x):
            return math.sin(x[0]) * math.cos(x[1])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        poles = bs.make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles)
        bs_plot.plot_surface_3d(surface_func, ax)
        bs_plot.plot_surface_poles_3d(surface_func, ax)

        plt.show()

    def test_evaluate(self):
        # self.plot_extrude()
        # self.plot_function()
        # TODO: test rational surfaces, e.g. sphere
        pass



class TestZ_Surface:

    # TODO: Compute max norm of the difference of two surfaces and assert that it is cose to zero.


    def plot_function_uv(self):
        # function surface
        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1]*4)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        poles = bs.make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles[:,:, [2] ])

        quad = np.array( [ [0, 0], [0, 0.5], [1, 0.1],  [1.1, 1.1] ]  )
        #quad = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        z_surf = bs.Z_Surface(quad, surface_func)
        full_surf = z_surf.make_full_surface()

        bs_plot.plot_surface_3d(z_surf, ax)
        bs_plot.plot_surface_3d(full_surf, ax, color='red')
        #bs_plot.plot_surface_poles_3d(surface_func, ax)

        plt.show()

    def test_eval_uv(self):
        #self.plot_function_uv()
        pass



class TestPointGrid:

    @staticmethod
    def function(x):
        return math.sin(x[0]) * math.cos(x[1])


    def make_point_grid(self):
        nu, nv = 5,6
        grid = bs.make_function_grid(TestPointGrid.function, 5, 6).reshape(nu*nv, 3)
        surf = bs.GridSurface(grid)
        return surf


    def plot_check_surface(self, XYZ_grid_eval, XYZ_surf_eval, XYZ_func_eval):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(XYZ_grid_eval[:, :, 0], XYZ_grid_eval[:, :, 1], XYZ_grid_eval[:, :, 2], color='blue')
        ax.plot_surface(XYZ_surf_eval[:, :, 0], XYZ_surf_eval[:, :, 1], XYZ_surf_eval[:, :, 2], color='red')
        ax.plot_surface(XYZ_func_eval[:, :, 0], XYZ_func_eval[:, :, 1], XYZ_func_eval[:, :, 2], color='green')
        plt.show()

    def grid_cmp(self, a, b, tol):
        a_z = a[:, :, 2].ravel()
        b_z = b[:, :, 2].ravel()
        eps = 0.0
        for i, (za, zb) in enumerate(zip(a_z, b_z)):
            diff = np.abs( za - zb)
            eps = max(eps, diff)
            assert diff < tol, " |a({}) - b({})| > tol({}), idx: {}".format(za, zb, tol, i)
        print("Max norm: ", eps, "Tol: ", tol)

    def check_surface(self, surf, xy_mat, xy_shift, z_mat):
        """
        TODO: Make this a general function - evaluate a surface on a grid, use it also in other tests
        to compare evaluation on the grid to the original function. Can be done after we have approximations.
        """

        nu, nv = 30, 40
        # surface on unit square
        U = np.linspace(0.0, 1.0, nu)
        V = np.linspace(0.0, 1.0, nv)
        V_grid, U_grid = np.meshgrid(V,U)

        UV = np.stack( [U_grid.ravel(), V_grid.ravel()], axis = 1 )
        XY = xy_mat.dot(UV.T).T + xy_shift
        Z = surf.z_eval_xy_array(XY)
        XYZ_grid_eval = np.concatenate( (XY, Z[:, None]) , axis = 1).reshape(nu, nv, 3)

        XYZ_surf_eval = surf.eval_array(UV).reshape(nu, nv, 3)

        z_func_eval = np.array([ z_mat[0]*TestPointGrid.function([u,v]) + z_mat[1]  for u, v in UV ])
        XYZ_func_eval = np.concatenate( (XY, z_func_eval[:, None]), axis =1 ).reshape(nu, nv, 3)

        #self.plot_check_surface(XYZ_grid_eval, XYZ_surf_eval, XYZ_func_eval)

        eps = 0.0
        hx = 1.0 / surf.shape[0]
        hy = 1.0 / surf.shape[1]
        tol = 0.5* ( hx*hx + 2*hx*hy + hy*hy)

        self.grid_cmp(XYZ_func_eval, XYZ_grid_eval, tol)
        self.grid_cmp(XYZ_func_eval, XYZ_surf_eval, tol)

    def test_grid_surface(self):
        xy_mat = np.array([ [1.0, 0.0], [0.0, 1.0] ])
        xy_shift = np.array([0.0, 0.0 ])
        z_shift = np.array([1.0, 0.0])
        surface = self.make_point_grid()

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # bs_plot.plot_grid_surface_3d(surface, ax)
        # plt.show()
        # self.check_surface(surface, xy_mat, xy_shift, z_shift)

        # transformed surface
        xy_mat = np.array([ [3.0, -3.0], [2.0, 2.0] ]) / math.sqrt(2)
        xy_shift = np.array([[-2.0, 5.0 ]])
        z_shift = np.array([1.0, 1.3])
        new_quad = np.array([ [0, 1.0], [0,0], [1, 0], [1, 1]])
        new_quad = new_quad.dot(xy_mat[:2,:2].T) + xy_shift

        surface = self.make_point_grid()
        surface.transform(np.concatenate((xy_mat, xy_shift.T), axis=1), z_shift)
        assert np.all(surface.quad == new_quad), "surf: {} ref: {}".format(surface.quad, new_quad)
        self.check_surface(surface, xy_mat, xy_shift, z_shift)


