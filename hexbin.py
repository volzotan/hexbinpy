from math import sqrt, pi, sin, cos, log, tan
from copy import deepcopy

import colormaps

import numpy as np
import cv2

class Hexbin():

    radius = 0.00001

    def __init__(self, diameter_in_metres):

        self.raw = []
        self.bins = {}

        self.radius = self._m_to_latlon(diameter_in_metres / 2.0)

        self.w = sqrt(3) * self.radius
        self.h = 2 * self.radius
        self.dx = self.w
        self.dy = 0.75 * self.h 


    def x(self, datapoint):
        return datapoint[0]


    def y(self, datapoint):
        return datapoint[1]


    def _m_to_latlon(self, m):
        return (m / 1.1) * 0.00001


    def add_data(self, data):
        for i in range(0, len(data)):

            px = self.x(data[i])
            py = self.y(data[i])

            py = py / self.dy
            pj = round(py)
            px = px / self.dx - (int(pj) & 1) / 2.0
            pi = round(px)
            py1 = py - pj;

            if (abs(py1) * 3 > 1):
                px1 = px - pi
                pi2 = pi + (-1 if px < pi else 1) / 2.0
                pj2 = pj + (-1 if py < pj else 1)
                px2 = px - pi2
                py2 = py - pj2
                if (px1 * px1 + py1 * py1 > px2 * px2 + py2 * py2):
                    pi = pi2 + (1 if int(pj) & 1 else -1) / 2.0
                    pj = pj2

            pi = int(pi)
            pj = int(pj)

            bin_id = "{}-{}".format(pi, pj)
            try:
                self.bins[bin_id][4] += 1
            except KeyError as e:
                binx = (pi + (pj & 1) / 2.0) * self.dx
                biny = pj * self.dy
                self.bins[bin_id] = [binx, biny, pi, pj, 1]

            self.raw.append([self.x(data[i]), self.y(data[i])])


    def _draw_hexagon(self, r):

        thirdPi = pi / 3
        angles = [0, thirdPi, 2 * thirdPi, 3 * thirdPi, 4 * thirdPi, 5 * thirdPi]

        x0 = 0
        y0 = 0

        res = []

        for angle in angles:
            x1 = sin(angle) * r
            y1 = -cos(angle) * r
            dx = x1 - x0
            dy = y1 - y0
            x0 = x1
            y0 = y1
            res.append([dx, dy])

        return res


    def _translate_hexagon(self, r, translation):
        coords = self._draw_hexagon(r)

        coords[0] = [coords[0][0] + translation[0], coords[0][1] + translation[1]]
        for i in range(1, len(coords)):
            coords[i] = [coords[i-1][0] + coords[i][0], coords[i-1][1] + coords[i][1]]

        return coords


    def _get_minmax(self):
        minval = None
        maxval = None

        for key, item in self.bins.items():
            val = item[4]

            if minval is None or val < minval:
                minval = val

            if maxval is None or val > maxval:
                maxval = val

        return (minval, maxval)


    def hexagons(self, translation=None):
        return self.bins


    @staticmethod
    def create_svg_path(points, absolute=False, inplace=False):
        cmd = []

        if inplace:
            cmd = points
        else:
            cmd = deepcopy(points)

        if absolute:
            cmd[0] = [["M"] + cmd[0]]
            cmd = cmd[0] + list(map(lambda x: ["L"] + x, cmd[1:]))
            cmd.append(["Z"])
        else:
            cmd[0] = [["m"] + cmd[0]]
            cmd = cmd[0] + list(map(lambda x: ["l"] + x, cmd[1:]))
            cmd.append(["z"])

        return cmd


    """
    Returns hexagon points with absolute coordinates. Information about bin fill values
    are stored separately in self.bins
    """
    def hexagon_points(self):
        points = []

        for key, item in self.bins.items():
            points.append(self._translate_hexagon(self.radius, translation=[item[0], item[1]]))

        return points


    """
    Returns SVG paths of hexagons, either with absolute coordinates or relative to (0,0).
    If non-absolute coordinates are used, the path object needs to be translated with the
    x,y information in hexbin.bins [0] and [1]
    """
    def hexagon_paths(self, absolute=True):
        paths = []

        for key, item in self.bins.items():
            if absolute:
                paths.append(create_svg_path(self._translate_hexagon(self.radius, translation=[item[0], item[1]]), absolute=True))
            else:
                paths.append(create_svg_path(self._draw_hexagon(self.radius)))

        return paths


    """
    Raw datapoints without binning.
    """
    def data(self):
        return self.raw


class Colorscale():

    def __init__(self, d):
        self.d = d

        from matplotlib.colors import ListedColormap
        self.colormap = ListedColormap(colormaps._viridis_data, name='viridis')
        # print(colormap(0.5)[:-1])

        # if type(value) in [list, tuple]:
        #     print("list")
        # else:
        #     print("no list")


    def get_color(self, value):
        a = value - self.d[0]

        if (a <= 0):
            return self.colormap(0)[:-1]

        a = a / (self.d[1] - self.d[0])

        if (a >= 1):
            return self.colormap(1.0)[:-1]

        return self.colormap(a)[:-1]


if __name__ == "__main__":
    print("INVOKE FROM SCRIPT, NOT DIRECTLY")

