from dataclasses import dataclass
from typing import Union, List

import numpy as np


@dataclass
class Point:
    """
    Dataclass to store point coordinates (cartesian).
    """

    x: Union[int, None]
    y: Union[int, None]

    def __add__(self, p2):
        return Point(self.x + p2.x, self.y + p2.y)

    def __sub__(self, p2):
        return Point(self.x - p2.x, self.y - p2.y)


@dataclass
class Line:
    """
    Data class to store line parameters
    y = ax + b

    a : slope
    b : intercept

    """
    intercept: float
    grad: float


    def __call__(self, x):
        return self.grad * x + self.intercept


@dataclass
class CapillaryPoints:
    """
    Points denoting a capillary object

    """
    p0: Point
    p1: Point
    p2: Point
    p3: Point
    p4: Point
    p5: Point
    p6: Point

    def get_point(self, idx):
        return self.__getattribute__(f'p{idx}')

    def __len__(self):
        return len(self.__dict__)

    def calc_distances(self, ref_point: Point = None) -> dict:
        """
        Calculate distances between each point of the capillary geometry with a reference point
        Parameters
        ----------
        ref_point: If None or empty, defaults to the p0 point as the reference.

        Returns
        -------

        """
        if ref_point is None:
            ref_point = self.p0
        distances = {}
        for idx in range(len(self.__dict__)):
            distances[f"d{idx}"] = dist_2D(self.get_point(idx), ref_point)
        return distances

    def project_points_to_lines(self, lines: List[Line]):
        projected_points = [project_point(line, self.get_point(idx)) for line, idx in zip(lines, range(len(self)))]
        return CapillaryPoints(*projected_points)


def get_intersection(line_1: Line, line_2: Line) -> Point:
    """
    Get intersection between two lines.

    Parameters
    ----------
    line_1
    line_2

    Returns
    -------
    Point object.
    """
    # intersection between top and bottom (from matlab)

    x = np.divide(line_2.intercept - line_1.intercept, line_1.grad - line_2.grad)
    y = line_1(x)

    return Point(x, y)


def dist_2D(p1: Point, p2: Point):
    """

    Parameters
    ----------
    p1 - Point object
    p2 - Point object

    Returns
    -------
    A value that reflects the distance between p1 and p2.

    """
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5


def get_tip_distances(capillary_points: List[CapillaryPoints]):
    return [caps.calc_distances() for caps in capillary_points]


def project_point(line: Line, p: Point) -> Point:
    """
    Projects a point on a line

    Parameters
    ----------
    line : Line object
    p : Point to project onto line

    Returns
    -------
    Projected point as a Point object.
    """
    if line is None:
        return p

    c_temp = (1 + line.grad ** 2) ** 0.5
    x_perp = -line.grad / c_temp
    y_perp = 1 / c_temp
    x = np.divide(p.y - line.intercept - (y_perp / x_perp) * p.x, line.grad - y_perp / x_perp)
    y = line(x)

    return Point(x, y)
