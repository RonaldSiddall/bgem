import enum
import numpy as np
from typing import *
from . import bspline

'''
TODO:
- For Solid make auto conversion from Faces similar to Face from Edges
- For Solid make test that Shells are closed.
- Implement closed test for shells (similar to wire)
- Improve test of slosed (Wire, Shell) to check also orientation of (Edges, Faces). ?? May be both if holes are allowed.
- Rename attributes and methods to more_words_are_separated_by_underscore.
- rename _writeformat to _brep_output
- remove (back) groups parameter from _brepo_output, make checks of IDs at main level (only in DEBUG mode)
- document public methods
'''


class ParamError(Exception):
    pass

def check_matrix(mat, shape, values, idx=tuple()):
    '''
    Check shape and type of scalar, vector or matrix.
    :param mat: Scalar, vector, or vector of vectors (i.e. matrix). Vector may be list or other iterable.
    :param shape: List of dimensions: [] for scalar, [ n ] for vector, [n_rows, n_cols] for matrix.
    If a value in this list is None, the dimension can be arbitrary. The shape list is set fo actual dimensions
    of the matrix.
    :param values: Type or tuple of  allowed types of elements of the matrix. E.g. ( int, float )
    :param idx: Internal. Used to pass actual index in the matrix for possible error messages.
    :return:
    TODO: replace check_matrix by conversion to appropriate numpy array.
    '''
    try:

        if len(shape) == 0:
            if not isinstance(mat, values):
                raise ParamError("Element at index {} of type {}, expected instance of {}.".format(idx, type(mat), values))
        else:

            if shape[0] is None:
                shape[0] = len(mat)
            l=None
            if not hasattr(mat, '__len__'):
                l=0
            elif len(mat) != shape[0]:
                l=len(mat)
            if not l is None:
                raise ParamError("Wrong len {} of element {}, should be  {}.".format(l, idx, shape[0]))
            for i, item in enumerate(mat):
                sub_shape = shape[1:]
                check_matrix(item, sub_shape, values, idx = (i, *idx))
                shape[1:] = sub_shape
        return shape
    except ParamError:
        raise
    except Exception as e:
        raise ParamError(e)


class BREPGroup(enum.IntEnum):
    locations = 0
    curves_3d = 1
    curves_2d = 2
    surfaces = 3
    shapes = 4


class BREPObject:
    """
    Basic class of the BREP objects, define common methods necessary for the
    file output. Objects forms a tree (or possibly DAG) and can be processed
    be a graph search without maintaining global structures.
    """
    def __init__(self, group : BREPGroup, id_in_postvisit=True) -> None:
        self._brep_group : BREPGroup = group
        self._brep_id : Optional[int] = None

        # Set ID and append to the BREP group in either DFS previsit or DFS postvisit.
        if id_in_postvisit:
            self._dfs_previsit = self._group_pass
            self._dfs_postvisit = self._group_append
        else:
            self._dfs_previsit = self._group_append
            self._dfs_postvisit = self._group_pass

    @property
    def brep_id(self):
        assert self._brep_id is not None, str(self)
        #if self._brep_id is  None:
        #    print("    None ID:", str(self))
        return self._brep_id


    def _childs(self):
        # Generator of the child BREPObjets for the DFS.
        # Default no childs.
        return []


    @staticmethod
    def gather_groups(objs):
        # DFS thorough the BREP object fromm the `self` as a root.
        # Assign BREP IDs and collect BREP objects into groups.
        visited = set()
        group_size = max(BREPGroup) + 1
        groups = [[] for _ in range(group_size)]
        for obj in objs:
            obj._dfs_gather_groups(groups, visited)
        return groups


    def _dfs_gather_groups(self, groups:List[List['BREPObject']], visited:Set[int]):
        # DFS recursive function.
        if id(self) in visited:
            return
        #print(f"visited: {self}")
        visited.add(id(self))

        self._dfs_previsit(groups)
        for ch in self._childs():
            ch._dfs_gather_groups(groups, visited)
        self._dfs_postvisit(groups)


    def _dfs_finish(self, visited:Set[int] = None):
        # DFS recursive function.
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))
        for ch in self._childs():
            ch._dfs_finish(visited)



    def _group_append(self, groups):
        #print(f"append: {id(self):x} {self} ")
        group = groups[self._brep_group]
        group.append(self)
        self._brep_id = len(group)



    def _group_pass(self, groups):
        pass



class Location(BREPObject):
    """
    Location defines an affine transformation in 3D space. Corresponds to the <location data 1> in the BREP file.
    BREP format allows to use different transformations for individual shapes.
    Location are numberd from 1. Zero index means identity location.

    TODO: possibly convert transformation methods to returning a Composed Location, that wy we
    can have a pure functional API.
    """

    @staticmethod
    def _identity_matrix():
        return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

    @staticmethod
    def Translate(vector):
        """
        Create a translation by the shift 'vector'.
        """
        matrix = Location._identity_matrix()
        matrix[:, 3] += np.array(vector, dtype=float)
        return Location(matrix)

    @staticmethod
    def Rotate(axis, angle, center=(0, 0, 0)):
        """
        Assuming the coordinate system:

        ^ Y
        |
        Z --> X

        Create a rotation anticlockwise (right hand rule) by the `angle` (radians)
        around the (normalised) `axis` vector.
        Optionally the center of the rotation can be specified.
        """
        matrix = Location._identity_matrix()
        center = np.array(center, dtype=float)
        axis = np.array(axis, dtype=float)
        axis /= np.linalg.norm(axis)

        W = np.array(
            [[0, -axis[2], axis[1]],
             [axis[2], 0, -axis[0]],
             [-axis[1], axis[0], 0]])
        M = np.eye(3) +  np.sin(angle) * W + 2 * np.sin(angle/2) ** 2 *  W @ W
        matrix[:, 3] -= center
        matrix = M @ matrix
        matrix[:, 3] += center
        return Location(matrix)

    @staticmethod
    def Scale(scale_vector, center=(0, 0, 0)):
        """
        Create a scaling the 'scale_vector' keeping 'center' unmodified.
        """
        matrix = Location._identity_matrix()
        center = np.array(center, dtype=float)
        scale_vector = np.array(scale_vector, dtype=float)
        matrix[:, 3] -= center
        matrix = np.diag(scale_vector) @ matrix
        matrix[:, 3] += center
        return Location(matrix)



    def __init__(self, matrix):
        """
        Constructor for elementary afine transformation.
        :param matrix: Transformation matrix 3x4. First three columns forms the linear transformation matrix.
        Last column is the translation vector.

        Location() is deprecated use 'Identity' object instead.

        TODO: Make matrix parameter obligatory.
        """
        super().__init__(group=BREPGroup.locations)
        if isinstance(matrix, str) and matrix == 'identity':
            return
        if matrix is None:
            print("Warning: 'Location()' is deprecated use 'Identity' instead.")
            self.matrix = None
            return

        # checks
        check_matrix(matrix, [3, 4], (int, float))
        self.matrix=np.array(matrix)


    # def _make_nondefault(self):
    #     if self.matrix is None:
    #         self.matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

    def apply(self, points:np.array) -> np.array:
        #  points: shape 3xN
        return self.matrix[:, :3] @ points + (self.matrix[:, 3])[:, None]

    def flat(self):
        return self

    def __matmul__(self, other: 'Location') -> 'ComposedLocation':
        """
        Can use matrix multiplication operator '@' to compose Locations.
        E.g.

        location = Identity @ Location.Rotate([0,1,0], angle) @ Location.Translate([1,2,3])

        is equivalent to

        location = Identity.rotate([0,1,0], angle).translate([1,2,3])

        Return ComposedLocation.
        """
        return ComposedLocation([(self, 1), (other,1)])

    def translate(self, vector):
        """
        Apply translation by the shift 'vector'.
        Return a composed location.
        """
        return self @ Location.Translate(vector)

    def rotate(self, axis, angle, center=(0, 0, 0)):
        """
        Apply rotation.
        Return a composed location.
        """
        return self @ Location.Rotate(axis, angle, center)

    def scale(self, scale_vector, center=(0, 0, 0)):
        """
        Apply scaling.
        Return a composed location.
        """
        return self @ Location.Scale(scale_vector, center)

    def _matrix_expand(self):
        return np.concatenate([self.matrix, np.array([[0,0,0,1]])], axis=0)

    def _brep_output(self, stream, groups):
        if self.matrix is None:
            # implicit identity
            return
        stream.write("1\n")
        for row in self.matrix:
            for number in row:
                stream.write(" {}".format(number))
            stream.write("\n")

class _Identity(Location):
    def __init__(self):
        super().__init__('identity')

    def _brep_output(self, stream, groups):
        pass
Identity = _Identity()


class ComposedLocation(Location):
    """
    Defines an affine transformation as a composition of othr transformations. Corresponds to the <location data 2> in the BREP file.
    BREP format allows to use different transformations for individual shapes.
    """
    def __init__(self, location_powers=None):
        """

        :param location_powers: List of pairs (location, power)  where location is instance of Location and power is float.
        """
        if location_powers is None:
            location_powers = []
        locs, pows =  zip(*location_powers)
        l = len(locs)
        check_matrix(locs, [ l ], (Location, ComposedLocation) )
        check_matrix(pows, [ l ], int)
        super().__init__('identity')
        self._locations = locs
        self._powers = pows

    def apply(self, points:np.array) -> np.array:
        #  points: shape 3xN
        return self.flat().apply(points)

    def flat(self):
        """
        Return the Location with the same matrix.
        """
        matrix = np.eye(4)
        for loc, pow in reversed(zip(self._locations, self._powers)):
            for _ in range(pow):
                full_matrix = loc._matrix_expand()
                matrix = full_matrix @ matrix
        return Location(matrix[:-1, :])



    def _childs(self):
        return self._locations


    def _brep_output(self, stream, groups):
        stream.write("2 ")
        for loc, pow in zip(self._locations, self._powers):
            stream.write("{} {} ".format(loc.brep_id, pow))
        stream.write("0\n")


def check_knots(deg, knots, N):
    total_multiplicity = 0
    for knot, mult in knots:
        # This condition must hold if we assume only (0,1) interval of curve or surface parameters.
        #assert float(knot) >= 0.0 and float(knot) <= 1.0
        total_multiplicity += mult
    assert total_multiplicity == deg + N + 1


# TODO: perform explicit conversion to np.float64 in order to avoid problems on different arch
# should be unified in bspline as well, convert  to np.arrays as soon as posible
scalar_types = (int, float, np.int32, np.int64, np.float32, np.float64)


def curve_from_bs( curve):
    """
    Make BREP writer Curve (2d or 3d) from bspline curve.
    :param curve: bs.Curve object
    :return:
    """
    dim = curve.dim
    if dim == 2:
        curve_dim = Curve2D
    elif dim == 3:
        curve_dim = Curve3D
    else:
        assert False
    c = curve_dim(curve.poles, curve.basis.pack_knots(), curve.rational, curve.basis.degree)
    c._bs_curve = curve
    return c

class Curve3D(BREPObject):
    """
    Defines a 3D curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <3D curve record 7>" from BREP format description.
    """
    def __init__(self, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline in 3d space.
        :param poles: List of poles (control points) ( X, Y, Z ) or weighted points (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                      degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Positive int.
        """

        if rational:
            check_matrix(poles, [None, 4], scalar_types )
        else:
            check_matrix(poles, [None, 3], scalar_types)
        N = len(poles)
        check_knots(degree, knots, N)

        super().__init__(group=BREPGroup.curves_3d)
        self.poles=poles
        self.knots=knots
        self.rational=rational
        self.degree=degree

    def _eval_check(self, t, point):
        if hasattr(self, '_bs_curve'):
            repr_pt = self._bs_curve.eval(t)
            if not np.allclose(np.array(point), repr_pt , rtol = 1.0e-3):
                raise Exception("Point: {} far from curve repr: {}".format(point, repr_pt))


    def _brep_output(self, stream, groups):
        # writes b-spline curve
        stream.write("7 {} 0  {} {} {} ".format(int(self.rational), self.degree, len(self.poles), len(self.knots)))
        for pole in self.poles:
            for value in pole:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in self.knots:
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")

class Curve2D(BREPObject):
    """
    Defines a 2D curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <2D curve record 7>" from BREP format description.
    """
    def __init__(self, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline in 2d space.
        :param poles: List of points ( X, Y ) or weighted points (X,Y, w). X,Y,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                      degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Positive int.
        """

        N = len(poles)
        if rational:
            check_matrix(poles, [N, 3], scalar_types )
        else:
            check_matrix(poles, [N, 2], scalar_types)
        check_knots(degree, knots, N)

        super().__init__(group=BREPGroup.curves_2d)

        self.poles=poles
        self.knots=knots
        self.rational=rational
        self.degree=degree

    def _eval_check(self, t, surface, point):
        if hasattr(self, '_bs_curve'):
            u, v = self._bs_curve.eval(t)
            surface._eval_check(u, v, point)


    def _brep_output(self, stream, groups):
        # writes b-spline curve
        stream.write("7 {} 0  {} {} {} ".format(int(self.rational), self.degree, len(self.poles), len(self.knots)))
        for pole in self.poles:
            for value in pole:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in self.knots:
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")



def surface_from_bs(surf):
    """
    Make BREP writer Surface from bspline surface.
    :param surf: bs.Surface object
    :return:
    """
    s = Surface(surf.poles, (surf.u_basis.pack_knots(), surf.v_basis.pack_knots()),
                    (surf.u_basis.degree, surf.v_basis.degree), surf.rational )
    s._bs_surface = surf
    return s

class Surface(BREPObject):
    """
    Defines a B-spline surface in 3d space. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Surface - < surface record 9 >" from BREP format description.
    """
    def __init__(self, poles, knots, degree=(2,2),  rational=False):
        """
        Construct a B-spline in 3d space.
        :param poles: Matrix (list of lists) of Nu times Nv poles (control points).
                      Single pole is a points ( X, Y, Z ) or weighted point (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: Tuple (u_knots, v_knots). Both u_knots and v_knots are lists of tuples
                      (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. For both U and V knot vector the total number of knots,
                      i.e. sum of their multiplicities, must be degree + N + 1, where N is number of poles.
        :param degree: (u_degree, v_degree) Both positive ints.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles. BREP format have two independent flags
                      for U and V parametr, but only choices 0,0 and 1,1 have sense.
        """

        if rational:
            check_matrix(poles, [None, None, 4], scalar_types )
        else:
            check_matrix(poles, [None, None, 3], scalar_types)

        assert len(poles) > 0
        assert len(poles[0]) > 0
        self.Nu = len(poles)
        self.Nv = len(poles[0])
        for row in poles:
            assert len(row) == self.Nv

        assert (not rational and len(poles[0][0]) == 3) or (rational and len(poles[0][0]) == 4)

        (u_knots, v_knots) = knots
        check_knots(degree[0], u_knots, self.Nu)
        check_knots(degree[1], v_knots, self.Nv)

        super().__init__(group=BREPGroup.surfaces)
        self.poles=poles
        self.knots=knots
        self.rational=rational
        self.degree=degree

    def _eval_check(self, u, v, point):
        if hasattr(self, '_bs_surface'):
            repr_pt = self._bs_surface.eval(u, v)
            if not np.allclose(np.array(point), repr_pt, rtol = 1.0e-3):
                raise Exception("Point: {} far from curve repr: {}".format(point, repr_pt))


    def _brep_output(self, stream, groups):
        #writes b-spline surface
        stream.write("9 {} {} 0 0 ".format(int(self.rational),int(self.rational))) #prints B-spline surface u or v rational flag - both same
        for i in self.degree: #prints <B-spline surface u degree> <_>  <B-spline surface v degree>
            stream.write(" {}".format(i))
        (u_knots, v_knots) = self.knots
        stream.write(" {} {}  {} {} ".format(self.Nu, self.Nv, len(u_knots), len(v_knots)))
            #prints  <B-spline surface u pole count> <_> <B-spline surface v pole count> <_> <B-spline surface u multiplicity knot count> <_>  <B-spline surface v multiplicity knot count> <B-spline surface v multiplicity knot count>
#        stream.write(" {}".format(self.poles)) #TODO: tohle smaz, koukam na format poles a chci: B-spline surface weight poles
        for pole in self.poles: #TODO: check, takovy pokus o poles
            for vector in pole:
                for value in vector:
                    stream.write(" {}".format(value))
                stream.write(" ")
            stream.write(" ")
        for knot in u_knots: #prints B-spline surface u multiplicity knots
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in v_knots: #prints B-spline surface v multiplicity knots
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")
            
class Approx:
    """
    Approximation methods for B/splines of degree 2.

    """
    @classmethod
    def plane(cls, vtxs):
        """
        Returns B-spline surface of a plane given by 3 points.
        We retun also list of UV coordinates of the given points.
        :param vtxs: List of tuples (X,Y,Z)
        :return: ( Surface, vtxs_uv )
        """
        assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
        vtxs.append( (0,0,0) )
        vtxs = np.array(vtxs)
        vv = vtxs[1] + vtxs[2] - vtxs[0]
        vtx4 = [ vtxs[0], vtxs[1], vv, vtxs[2]]
        (surf, vtxs_uv) = cls.bilinear(vtx4)
        return (surf, [ vtxs_uv[0], vtxs_uv[1], vtxs_uv[3] ])

    @classmethod
    def bilinear(cls, vtxs):
        """
        Returns B-spline surface of a bilinear surface given by 4 corner points.
        We retun also list of UV coordinates of the given points.
        :param vtxs: List of tuples (X,Y,Z)
        :return: ( Surface, vtxs_uv )
        """
        assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
        vtxs = np.array(vtxs)
        def mid(*idx):
            return np.mean( vtxs[list(idx)], axis=0)

        # v - direction v0 -> v2
        # u - direction v0 -> v1
        poles = [ [vtxs[0],  mid(0, 3), vtxs[3]],
                  [mid(0,1), mid(0,1,2,3), mid(2,3)],
                  [vtxs[1], mid(1,2), vtxs[2]]
                  ]
        knots = [(0.0, 3), (1.0, 3)]
        bs_surface = bspline.Surface.make_raw(poles, (knots, knots), degree=(2,2), rational=False)
        surface = surface_from_bs(bs_surface)
        vtxs_uv = [ (0, 0), (1, 0), (1, 1), (0, 1) ]
        return (surface, vtxs_uv)




    @classmethod
    def _line(cls, vtxs, overhang=0.0):
        '''
        :param vtxs: List of tuples (X,Y) or (X,Y,Z)
        :return:
        '''
        assert len(vtxs) == 2
        vtxs = np.array(vtxs)
        mid = np.mean(vtxs, axis=0)
        poles = [ vtxs[0],  mid, vtxs[1] ]
        knots = [(0.0+overhang, 3), (1.0-overhang, 3)]
        return (poles, knots)

    @classmethod
    def line_2d(cls, vtxs):
        """
        Return B-spline approximation of line from two 2d points
        :param vtxs: [ (X0, Y0), (X1, Y1) ]
        :return: Curve2D
        """
        return Curve2D( *cls._line(vtxs) )

    @classmethod
    def line_3d(cls,  vtxs):
        """
        Return B-spline approximation of line from two 3d points
        :param vtxs: [ (X0, Y0, Z0), (X1, Y1, Z0) ]
        :return: Curve2D
        """
        return Curve3D(*cls._line(vtxs))


class Orient(enum.IntEnum):
    Forward=1
    Reversed=2
    Internal=3
    External=4

#op=Orient.Forward
#om=Orient.Reversed
#oi=Orient.Internal
#oe=Orient.External

class ShapeRef:
    """
    Auxiliary data class to store an object with its orientation
    and possibly location. Meaning of location in this context is not clear yet.
    Identity location (0) in all BREPs produced by OCC.
    All methods accept the tuple (shape, orient, location) and
    construct the ShapeRef object automatically.
    """

    orient_chars = ['+', '-', 'i', 'e']

    def __init__(self, shape, orient=Orient.Forward, location=Identity):
        """
        :param shape: referenced shape
        :param orient: orientation of the shape, value is enum Orient
        :param location: A Location object. Default is None = identity location.
        """
        if not issubclass(type(shape), Shape):
            raise ParamError("Expected Shape, get: {}.".format(shape))
        assert isinstance(orient, Orient)
        assert issubclass(type(location), Location)

        self.shape=shape
        self.orientation=orient
        self.location=location

    def _writeformat(self, stream, groups):

        stream.write("{}{} {} ".format(self.orient_chars[self.orientation-1], self.shape._brep_id, self.location._brep_id))

    def __repr__(self):
        return "{}{} ".format(self.orient_chars[self.orientation-1], self.shape._brep_id)


class ShapeFlag(dict):
    """
    Auxiliary data class representing the shape flag word of BREP shapes.
    All methods set the flags automatically, but it can be overwritten.

    Free - Seems to indicate a top level shapes.
    Modified - ??
    Checked - for format version 2 may indicate that shape topology is already checked
    Orientable - ??
    Closed - used to indicate closed Wires and Shells
    Infinite - ?? may indicate shapes extending to infinite, not our case
    Convex - ?? may indicate convexity of the shape, not clear how this is combined with geometry
    """
    flag_names = ['free', 'modified', 'checked', 'orientable', 'closed', 'infinite', 'convex']

    def __init__(self, *args):
        for k, f in zip(self.flag_names, args):
            assert f in [0, 1]
            self[k]=f

    def set(self, key, value=1):
        if value:
            value =1
        else:
            value =0
        self[key] = value

    def unset(self, key):
        self[key] = 0

    def _brep_output(self, stream):
        for k in self.flag_names:
            stream.write(str(self[k]))

class Shape(BREPObject):
    def __init__(self, childs):
        """
        Construct base Shape object.
        Examples:
            Wire([ edge_1, edge_2.m(), edge_3])     # recommended
            Wire(edge_1, ShapeRef(edge_2, Orient.Reversed, some_location), edge_3)
            ... not recommended since it is bad idea to reference same shape with different Locations.

        :param childs: List of ShapeRefs or child objects.
        """

        # self.subtypes - List of allowed types of childs.
        assert hasattr(self, 'sub_types'), self

        # convert list of shape reference tuples to ShapeRef objects
        # automaticaly wrap naked shapes into tuple.
        self.childs=[]
        for child in childs:
            self.append(child)   # append convert to ShapeRef

        # Thes flags are usualy produced by OCC for all other shapes safe vertices.
        self.flags=ShapeFlag(0,1,0,1,0,0,0)

        super().__init__(group=BREPGroup.shapes)
        assert hasattr(self, 'brep_shpname'),  self
        # Name of particular shape in BREP format, defined in childs.
        assert hasattr(self, 'sub_types')
        # Valid types of the shape childs.

    def _childs(self):
        for sub_ref in self.childs:
            yield sub_ref.location
            yield sub_ref.shape
    """
    Methods to simplify ceration of oriented references.
    """
    def p(self):
        return ShapeRef(self, Orient.Forward)

    def m(self):
        return ShapeRef(self, Orient.Reversed)

    def i(self):
        return ShapeRef(self, Orient.Internal)

    def e(self):
        return ShapeRef(self, Orient.External)

    def subshapes(self):
        # Return list of subshapes stored in child ShapeRefs.
        return [chld.shape for chld in self.childs]

    def append(self, shape_ref):
        """
        Append a reference to shild
        :param shape_ref: Either ShapeRef or child shape.
        :return: None
        """
        if type(shape_ref) != ShapeRef:
            shape_ref=ShapeRef(shape_ref)
        if not isinstance(shape_ref.shape, tuple(self.sub_types)):
            raise ParamError("Wrong child type: {}, allowed: {}".format(type(shape_ref.shape), self.sub_types))
        self.childs.append(shape_ref)

    #def _convert_to_shaperefs(self, childs):


    def set_flags(self, flags):
        """
        Set flags given as tuple.
        :param flags: Tuple of 7 flags.
        :return:
        """
        self.flags = ShapeFlag(*flags)


    def is_closed(self):
        return self.flags['closed']


    def _brep_output(self, stream, groups):
        stream.write("{}\n".format(self.brep_shpname))
        self._subrecordoutput(stream)
        self.flags._brep_output(stream)
        stream.write("\n")
#        stream.write("{}".format(self.childs))
        for child in self.childs:
            child._writeformat(stream, groups)
        stream.write("*\n")
        #subshape, tj. childs

    def _subrecordoutput(self, stream):
        stream.write("\n")

    def _head(self):
        return f"{id(self):x} {self.brep_shpname} {str(self._brep_id)} "


    def __repr__(self):
        #if not hasattr(self, 'id'):
        #    self.index_all()
        repr = self._head()
        #if len(self.childs)==0:
        #    return ""
        repr += " : ["
        for child in self.childs:
            repr += child.shape._head()
        repr += "]"
        repr += "\n"
        return repr


"""
Shapes with no special parameters, only flags and subshapes.
Writer can be generic implemented in bas class Shape.
"""

class Compound(Shape):
    def __init__(self, shapes=None):
        if shapes is None:
            shapes = []
        self.sub_types =  [CompoundSolid, Solid, Shell, Wire, Face, Edge, Vertex]
        self.brep_shpname = 'Co'
        super().__init__(shapes)
        #flags: free, modified, IGNORED, orientable, closed, infinite, convex
        self.set_flags( (1, 1, 0, 0, 0, 0, 0) ) # free, modified

    def set_free_shapes(self):
        """
        Set 'free' attributes to all shapes of the compound.
        :return:
        """
        for shape in self.subshapes():
            shape.flags.set('free', True)


class CompoundSolid(Shape):
    def __init__(self, solids=None):
        self.sub_types = [Solid]
        self.brep_shpname = 'Cs'
        super().__init__(solids)


class Solid(Shape):
    def __init__(self, shells=None):
        self.sub_types = [Shell]
        self.brep_shpname='So'
        super().__init__(shells)
        self.set_flags((0, 1, 0, 0, 0, 0, 0))  # modified

class Shell(Shape):
    def __init__(self, faces=None):
        self.sub_types = [Face]
        self.brep_shpname='Sh'
        super().__init__(faces)
        self.set_flags((0, 1, 0, 1, 0, 0, 0))  # modified, orientable


class Wire(Shape):
    def __init__(self, edges=None):
        self.sub_types = [Edge]
        self.brep_shpname='Wi'
        super().__init__(edges)
        self.set_flags((0, 1, 0, 1, 0, 0, 0))  # modified, orientable
        self._set_closed()

    def _set_closed(self):
        '''
        Return true for the even parity of vertices.
        :return: REtrun true if wire is closed.
        '''
        vtx_set = {}
        for edge in self.subshapes():
            for vtx in edge.subshapes():
                vtx_set[vtx] = 0
                vtx.n_edges += 1
        closed = True
        for vtx in vtx_set.keys():
            if vtx.n_edges % 2 != 0:
                closed = False
            vtx.n_edges = 0
        self.flags.set('closed', closed)


"""
Shapes with special parameters.
Specific writers are necessary.
"""

class Face(Shape):
    """
    Face class.
    Like vertex and edge have some additional parameters in the BREP format.
    """

    def __init__(self, wires, surface=None, location=Identity, tolerance=1.0e-3):
        """
        :param wires: List of wires, or list of edges, or list of ShapeRef tuples of Edges to construct a Wire.
        :param surface: Representation of the face, surface on which face lies.
        :param location: Location of the surface.
        :param tolerance: Tolerance of the representation.
        """
        self.sub_types = [Wire, Edge]
        self.tol=tolerance
        self.restriction_flag =0
        self.brep_shpname = 'Fa'

        if type(wires) != list:
            wires = [ wires ]
        assert(len(wires) > 0)
        super().__init__(wires)

        # auto convert list of edges into wire
        shape_type = type(self.childs[0].shape)
        for shape in self.subshapes():
            assert type(shape) == shape_type
        if shape_type == Edge:
            wire = Wire(self.childs)
            self.childs = []
            self.append(wire)

        # check that wires are closed
        for wire in self.subshapes():
            if not wire.is_closed():
                raise Exception("Trying to make face from non-closed wire.")

        if surface is None:
            self.repr=[]
        else:
            assert type(surface) == Surface
            self.repr=[(surface, location)]

    def _childs(self):
        # Finalize the shape.
        assert len(self.repr) == 1

        for repr, loc in self.repr:
            yield repr
            yield loc

        yield from super()._childs()

    def _dfs_finish(self, visited):
        if not self.repr:
            self.implicit_surface()
        super(Face, self)._dfs_finish(visited)

    def implicit_surface(self):
        """
        Construct a surface if surface is None. Works only for
        3 and 4 vertices (plane or bilinear surface)
        Should be called in _dfs just after all child shapes are passed.
        :return: None

        TODO: simplify
        """
        edges = {}
        vtxs = []
        for wire in self.subshapes():
            for edge in wire.childs:
                edges[id(edge.shape)] =  edge.shape
                e_vtxs = edge.shape.subshapes()
                if edge.orientation == Orient.Reversed:
                    e_vtxs.reverse()
                for vtx in e_vtxs:
                    vtxs.append( (id(vtx), vtx.point) )
        vtxs = vtxs[1:] + vtxs[:1]
        odd_vtx = vtxs[1::2]
        even_vtx = vtxs[0::2]
        assert odd_vtx == even_vtx, "odd: {} even: {}".format(odd_vtx, even_vtx)
        vtxs = odd_vtx
        if len(vtxs) == 3:
            constructor = Approx.plane
        elif len(vtxs) == 4:
            constructor = Approx.bilinear
        else:
            raise Exception("Too many vertices {} for implicit surface construction.".format(len(vtxs)))
        (ids, points) = zip(*vtxs)
        (surface, vtxs_uv) =  constructor(list(points))
        self.repr = [(surface, Identity)]

        # set representation of edges
        assert len(ids) == len(vtxs_uv)
        id_to_uv = dict(zip(ids, vtxs_uv))
        for edge in edges.values():
            e_vtxs = edge.subshapes()
            v0_uv = id_to_uv[id(e_vtxs[0])]
            v1_uv = id_to_uv[id(e_vtxs[1])]
            edge.attach_to_surface(surface, v0_uv, v1_uv)

        # TODO: Possibly more general attachment of edges to 2D curves for general surfaces, but it depends
        # on organisation of intersection curves.
        return self

    def _subrecordoutput(self, stream):
        assert len(self.repr) == 1
        surf,loc = self.repr[0]
        stream.write("{} {} {} {}\n\n".format(self.restriction_flag, self.tol, surf.brep_id, loc.brep_id))


class Edge(Shape):
    """
    Edge class. Special edge flags have unclear meaning.
    Allow setting representations of the edge, this is crucial for good mash generation.
    """

    class Repr(enum.IntEnum):
        Curve3d = 1
        Curve2d = 2
        #Continuous2d=3


    def __init__(self, vertices, tolerance=1.0e-3):
        """
        :param vertices: List of shape reference tuples, see ShapeRef class.
        :param tolerance: Tolerance of the representation.
        """
        self.sub_types = [Vertex]
        self.brep_shpname = 'Ed'
        self.tol = tolerance
        self.repr = []
        self.edge_flags=(1,1,0)         # this is usual value

        assert(len(vertices) == 2)

        super().__init__(vertices)
        # Overwrite vertex orientation
        self.childs[0].orientation = Orient.Forward
        self.childs[1].orientation = Orient.Reversed

    def set_edge_flags(self, same_parameter, same_range, degenerated):
        """
        Edge flags with unclear meaning.
        :param same_parameter:
        :param same_range:
        :param degenerated:
        :return:
        """
        self.edge_flags=(same_parameter, same_range, degenerated)

    def points(self):
        '''
        :return: List of coordinates of the edge vertices.
        '''
        return [ vtx.point for vtx in self.subshapes()]

    def attach_to_3d_curve(self, t_range, curve, location=Identity):
        """
        Add vertex representation on a 3D curve.
        :param t_range: Tuple (t_min, t_max).
        :param curve: 3D curve object (Curve3d)
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        assert type(curve) == Curve3D
        curve._eval_check(t_range[0], self.points()[0])
        curve._eval_check(t_range[1], self.points()[1])
        self.repr.append( (self.Repr.Curve3d, t_range, curve, location) )
        return self

    def attach_to_2d_curve(self, t_range, curve, surface, location=Identity):
        """
        Add vertex representation on a 2D curve.
        :param t_range: Tuple (t_min, t_max).
        :param curve: 2D curve object (Curve2d)
        :param surface: Surface on which the curve lies.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        #print(f"attach: {self} {curve}")
        assert type(surface) == Surface
        assert type(curve) == Curve2D
        curve._eval_check(t_range[0], surface, self.points()[0])
        curve._eval_check(t_range[1], surface, self.points()[1])
        self.repr.append( (self.Repr.Curve2d, t_range, curve, surface, location) )
        return self

    def _vtx_surface_uv(self, i, surface, uv_vtx=None):
        if uv_vtx is not None:
            return uv_vtx
        return self.childs[i].shape._surface_uv(surface)

    def attach_to_surface(self, surface, v0=None, v1=None):
        """
        Construct and attach 2D line in UV space of the 'surface'
        :param surface: A Surface object.
        :param v0: UV coordinate of the first edge point
        :param v1: UV coordinate of the second edge point
        Try to get UV coordinates from the end points.

        :return:
        """
        assert type(surface) == Surface
        v0 = self._vtx_surface_uv(0, surface, v0)
        v1 = self._vtx_surface_uv(1, surface, v1)
        self.attach_to_2d_curve((0.0, 1.0), Approx.line_2d([v0, v1]), surface)
        return self

    def implicit_curve(self):
        """
        Construct a line 3d curve if there is no 3D representation.
        Should be called in _dfs.
        :return:
        """
        vtx_points = self.points()
        self.attach_to_3d_curve((0.0,1.0), Approx.line_3d( vtx_points ))
        return self

    def _dfs_finish(self, visited):
        if all( (r[0] != self.Repr.Curve3d for r in self.repr) ):
            self.implicit_curve()
        # No need to finish Vertex

    def _childs(self):
        # finalize
        assert len(self.repr) > 0

        for repr in self.repr:
            if repr[0] == self.Repr.Curve2d:
                yield repr[2]
                yield repr[3]
                yield repr[4]
            elif repr[0] == self.Repr.Curve3d:
                yield repr[2]
                yield repr[3]
        yield from super()._childs()


    def _subrecordoutput(self, stream):
        #print(f"subrecord: {self} {id(self):x}")
        assert len(self.repr) > 0
        stream.write(" {} {} {} {}\n".format(self.tol,self.edge_flags[0],self.edge_flags[1],self.edge_flags[2]))
        for i,repr in enumerate(self.repr):
            if repr[0] == self.Repr.Curve2d:
                curve_type, t_range, curve, surface, location = repr
                stream.write("2 {} {} {} {} {}\n".format(
                    curve.brep_id, surface.brep_id, location.brep_id, t_range[0],t_range[1] ))

            elif repr[0] == self.Repr.Curve3d:
                curve_type, t_range, curve, location = repr
                stream.write("1 {} {} {} {}\n".format(curve.brep_id, location.brep_id, t_range[0], t_range[1]))
        stream.write("0\n")


class Vertex(Shape):
    """
    Vertex class.
    Allow setting representations of the vertex but seems it is not used in BREPs produced by OCC.
    """

    class Repr(enum.IntEnum):
        Curve3d = 1
        Curve2d = 2
        Surface = 3

    @staticmethod
    def on_surface(u, v, surface, location=Identity):
        point = surface._bs_surface.eval(u, v)
        return Vertex(point).attach_to_surface(u, v, surface, location)

    @staticmethod
    def on_curve_2d(t, curve, surface, location=Identity):
        uv = curve._bs_curve.eval(t)
        point = surface._bs_surface.eval(*uv)
        return Vertex(point).attach_to_2d_curve(t, curve, surface, location)

    @staticmethod
    def on_curve_3d(t, curve, location=Identity):
        point = curve._bs_curve.eval(t)
        return Vertex(point).attach_to_3d_curve(t, curve, location)


    def __init__(self, point, tolerance=1.0e-3):
        """
        :param point: 3d point (X,Y,Z)
        :param tolerance: Tolerance of the representation.
        """
        check_matrix(point, [3], scalar_types)

        # These flags are produced by OCC for vertices.
        self.flags = ShapeFlag(0, 1, 0, 1, 1, 0, 1)
        # Coordinates in the 3D space. [X, Y, Z]
        self.point=np.array(point)
        # tolerance of representations.
        self.tolerance=tolerance
        # List of geometrical representations of the vertex. Possibly not necessary for meshing.
        self.repr=[]
        # Number of edges in which vertex is used. Used internally to check closed wires.
        self.n_edges = 0
        self.brep_shpname = 'Ve'
        self.sub_types=[]

        super().__init__(childs=[])

    def attach_to_3d_curve(self, t, curve, location=Identity):
        """
        Add vertex representation on a 3D curve.
        :param t: Parameter of the point on the curve.
        :param curve: 3D curve object (Curve3d)
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        curve._eval_check(t, self.point)
        self.repr.append( (self.Repr.Curve3d, t, curve, location) )
        return self

    def attach_to_2d_curve(self, t, curve, surface, location=Identity):
        """
        Add vertex representation on a 2D curve on a surface.
        :param t: Parameter of the point on the curve.
        :param curve: 2D curve object (Curve2d)
        :param surface: Surface on which the curve lies.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        curve._eval_check(t, surface, self.point)
        self.repr.append( (self.Repr.Curve2d, t, curve, surface, location) )
        return self

    def attach_to_surface(self, u, v, surface, location=Identity):
        """
        Add vertex representation on a 3D curve.
        :param u,v: Parameters u,v  of the point on the surface.
        :param surface: Surface object.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        surface._eval_check(u, v, self.point)
        self.repr.append( (self.Repr.Surface, u,v, surface, location) )
        return self

    def _childs(self):
        for repr in self.repr:
            if repr[0]==self.Repr.Surface:
                yield repr[3] #surface
                yield repr[4] #location
            if repr[0]==self.Repr.Curve2d:
                yield repr[2] #curve
                yield repr[3] #surface
                yield repr[4] #location
            elif repr[0]==self.Repr.Curve3d:
                yield repr[2] #curve
                yield repr[3] #location
        yield from super()._childs()


    def _subrecordoutput(self, stream): #prints vertex data
        stream.write("{}\n".format(self.tolerance))
        for i in self.point:
            stream.write("{} ".format(i))
        stream.write("\n")

        # <vertex data representation>
        for i,repr in enumerate(self.repr):
            if repr[0] == self.Repr.Surface:
                _, u, v, surface, location = repr
                stream.write("3 {} {} {} {}\n".format(
                    u, v, surface.brep_id, location.brep_id))
            if repr[0] == self.Repr.Curve2d:
                _, t, curve, surface, location = repr
                stream.write("2 {} {} {} {}\n".format(
                    t, curve.brep_id, surface.brep_id, location.brep_id))

            elif repr[0] == self.Repr.Curve3d:
                _, t, curve, location = repr
                stream.write("1 {} {} {}\n".format(
                    t, curve.brep_id, location.brep_id))

        stream.write("\n0 0\n\n")

    def _surface_uv(self, surface):
        for r in self.repr:
            if r[0] == self.Repr.Surface and r[3] is surface:
                return (r[1], r[2])
        raise KeyError("Vertex not attached to the surface.")

    # def _curve_t(self, curve):
    #     for r in self.repr:
    #         if r[0] == self.Repr.Surface and r[3] is surface:
    #             return (r[1], r[2])
    #     raise KeyError("Vertex not attached to the surface.")


def write_model(stream, compound, location=Identity):
    """
    Write a BREP representation of the model 'compound' transformed to the 'location'
    to the 'stream'.
    """
    assert isinstance(compound, Compound)
    compound._dfs_finish()
    groups = BREPObject.gather_groups([Identity, compound, location])
    locations = groups[BREPGroup.locations]
    curves_3d = groups[BREPGroup.curves_3d]
    curves_2d = groups[BREPGroup.curves_2d]
    surfaces = groups[BREPGroup.surfaces]
    shapes = groups[BREPGroup.shapes]

    # modify IDs according to the BREP format
    for loc in locations:
        loc._brep_id -= 1
    n_shapes = len(shapes) + 1
    for shape in shapes:
        shape._brep_id = n_shapes - shape._brep_id

    stream.write("DBRep_DrawableShape\n\n")
    stream.write("CASCADE Topology V1, (c) Matra-Datavision\n")
    stream.write("Locations {}\n".format(len(locations) - 1))
    for loc in locations[1:]:
        loc._brep_output(stream, groups)

    stream.write("Curve2ds {}\n".format(len(curves_2d)))
    for curve in curves_2d:
        curve._brep_output(stream, groups)

    stream.write("Curves {}\n".format(len(curves_3d)))
    for curve in curves_3d:
        curve._brep_output(stream, groups)

    stream.write("Polygon3D 0\n")

    stream.write("PolygonOnTriangulations 0\n")

    stream.write("Surfaces {}\n".format(len(surfaces)))
    for surface in surfaces:
        surface._brep_output(stream, groups)

    stream.write("Triangulations 0\n")

    stream.write("\nTShapes {}\n".format(len(shapes)))
    for shape in shapes:
        shape._brep_output(stream, groups)
    stream.write(f"\n+1 {location.brep_id}")


