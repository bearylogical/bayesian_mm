from src.utils.geometry import Point, Line, CapillaryPoints
import numpy as np


class TestCapillaryGeometry:

    def test_point_dataclass(self):
        mock_point_1 = Point(1, 2)
        mock_point_2 = Point(0, 1)
        assert mock_point_1 == Point(1, 2)
        assert mock_point_1 + mock_point_2 == Point(1, 3)
        assert mock_point_1 - mock_point_2 == Point(1, 1)
        assert mock_point_2 - mock_point_1 == Point(-1, -1)

    def test_line_dataclass(self):
        line_1 = Line(0, 1)
        arr_x = np.array([1, 2, 3, 4, 5])

        assert line_1(1) == 1
        assert line_1(arr_x) == np.array([1, 1, 1, 1, 1])

    def test_capillary_dataclass(self):
        mock_points = []
        for i in range(7):
            mock_points.append(Point(0, i))

        mock_capillary_point = CapillaryPoints(*mock_points)
        res_distance = mock_capillary_point.calc_distances()

        assert mock_capillary_point == CapillaryPoints(Point(0, 0), Point(0, 1),
                                                       Point(0, 2), Point(0, 3), Point(0, 4),
                                                       Point(0, 5), Point(0, 6))

        assert res_distance == dict(p0=0., p1=1., p2=2., p3=3., p4=4., p5=5., p6=6.)
