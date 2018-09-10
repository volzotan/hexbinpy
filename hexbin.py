from math import sqrt, pi, sin, cos
from copy import deepcopy
import svgwrite

import cv2

CSS_FILENAME = "style.css"

class Hexbin():

    radius = 1

    def __init__(self, data, radius):

        self.raw = []
        self.bins = {}

        self.radius = radius

        self.w = sqrt(3) * self.radius
        self.h = 2 * self.radius
        self.dx = self.w
        self.dy = 0.75 * self.h 

        self._add_data(data)


    def _add_data(self, data):
        for i in range(0, len(data)):

            px = data[i][0]
            py = data[i][1]

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

            bin_id = "{}-{}".format(pi, pj)
            try:
                self.bins[bin_id][4] += 1
            except KeyError as e:
                binx = (pi + (int(pj) & 1) / 2.0) * self.dx
                biny = pj * self.dy
                self.bins[bin_id] = [binx, biny, pi, pj, 1]

            self.raw.append(data[i])


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


    def _interpolate_color(self, value, domain, range):
        return (0, 0, 0)


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


def transform_points():
    pass


def save_svg(filename, hexagons, points, dimensions=None, image=None):

    if not filename.endswith(".svg"):
        filename += ".svg"

    dwg = None

    if dimensions is not None:
        size = (str(dimensions[0]) + "px", str(dimensions[1]) + "px")
        dwg = svgwrite.Drawing(filename, profile="tiny", size=size)
    else:
        dwg = svgwrite.Drawing(filename, profile="tiny")

    if image is not None:
        dwg.add(svgwrite.image.Image(image, insert=(0, 0)))

    dwg.add_stylesheet(CSS_FILENAME, title="main_stylesheet")

    for item in hexagons:
        # dwg.add(dwg.path(self._get_hexagon_path(self.radius), transform="translate({},{})".format(item[0], item[1]), fill="rgb(254,0,0)"))
        dwg.add(
            dwg.path(Hexbin.create_svg_path(item, absolute=True))
        ) 

        # , 
        #     fill=svgwrite.rgb(0, 0, 0),
        #     stroke=svgwrite.rgb(0, 0, 0)
        # )

         # fill=svgwrite.rgb(*self._interpolate_color(item[4], self._get_minmax(), ((10, 10, 10), (200, 200, 200)))), 

    # for item in points:
    #     dwg.add(dwg.circle(center=item, r=0.5, fill=svgwrite.rgb(254, 0, 0)))

    dwg.save()


if __name__ == "__main__":

    example_data = [
        [1, 1],
        [5, 1],
        [1, 10],
        [5, 10],
        [50, 2],
        [3, 70],
        [80, 75]
    ]

    # example_data = [
    #     # [1, 2],
    #     [1, 12]
    # ]

    # import numpy as np 
    # mu, sigma = 0, 0.3  
    # n = 1000
    # x = np.multiply(np.random.normal(mu, sigma, n), 400)
    # y = np.multiply(np.random.normal(mu, sigma, n), 300)
    # x = np.add(x, 400)
    # y = np.add(y, 300)
    # example_data = [None] * n
    # for i in range(0, n):
    #     example_data[i] = [float(x[i]), float(y[i])]

    import csv

    example_data = []
    with open("boxes_bahnhof2.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        data = [r for r in csv_reader]
        data.pop(0) # remove header
        
        for line in data:
            minx = float(line[6])
            miny = float(line[7])
            maxx = float(line[8])
            maxy = float(line[9])

            example_data.append([minx + (maxx-minx)/2, maxy])

    print(len(example_data))

    example_data = example_data[0:100]

    hexbin = Hexbin(example_data, 20)
    # hexbin.homography()
    save_svg("text.svg", hexbin.hexagon_points(), hexbin.data(), image="bg.jpg", dimensions=(5952, 3348))

    print(hexbin._get_minmax())

    # for key, item in hexbin.hexagons().items():
    #     print(item)


