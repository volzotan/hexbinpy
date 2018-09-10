from math import sqrt, pi, sin, cos, log, tan
from copy import deepcopy
import svgwrite

import colormaps

import numpy as np
import cv2

CSS_FILENAME = "style.css"

class Hexbin():

    radius = 0.00001

    def __init__(self, radius_in_metres):

        self.raw = []
        self.bins = {}

        self.radius = self._m_to_latlon(radius_in_metres)

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

            bin_id = "{}-{}".format(pi, pj)
            try:
                self.bins[bin_id][4] += 1
            except KeyError as e:
                binx = (pi + (int(pj) & 1) / 2.0) * self.dx
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


class Writer():

    def __init__(self, filename, dimensions=None, image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename = filename
        self.dwg = None

        if dimensions is not None:
            size = (str(dimensions[0]) + "px", str(dimensions[1]) + "px")
            self.dwg = svgwrite.Drawing(filename, profile="tiny", size=size)
        else:
            self.dwg = svgwrite.Drawing(filename, profile="tiny")

        if image is not None:
            self.dwg.add(svgwrite.image.Image(image, insert=(0, 0)))

        self.dwg.add_stylesheet(CSS_FILENAME, title="main_stylesheet")


    def add_hexagons(self, hexagons, fills):
        for i in range(0, len(hexagons)):
            # dwg.add(dwg.path(self._get_hexagon_path(self.radius), transform="translate({},{})".format(item[0], item[1]), fill="rgb(254,0,0)"))
            self.dwg.add(self.dwg.path(
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                fill=svgwrite.rgb(fills[i][0]*254, fills[i][1]*254, fills[i][2]*254))
            ) 

            # , 
            #     fill=svgwrite.rgb(0, 0, 0),
            #     stroke=svgwrite.rgb(0, 0, 0)
            # )

             # fill=svgwrite.rgb(*self._interpolate_color(item[4], self._get_minmax(), ((10, 10, 10), (200, 200, 200)))), 

    def add_circles(self, points, radius=3):
        for item in points:
            self.dwg.add(self.dwg.circle(center=item, r=radius, fill=svgwrite.rgb(254, 0, 0)))


    def save(self):
        self.dwg.save()



def calculateHomographyMatrix(points_src, points_dst):

    pts_src = np.array(points_src, dtype=np.float)
    pts_dst = np.array(points_dst, dtype=np.float)

    h, status = cv2.findHomography(pts_src, pts_dst)

    return h, status


def transformPoints(points, h):
    warped = []

    for point in points:
        values = np.array([[point]], dtype=np.float64)
        pointsOut = cv2.perspectiveTransform(values, h)
        warped.append(pointsOut.tolist()[0][0])

    return warped


def transformHexagons(hexagons, h):
    warped_hexagons = []

    for hexagon in hexagons:
        warped_hexagon = []
        for point in hexagon:
            values = np.array([[point]], dtype=np.float64)
            pointsOut = cv2.perspectiveTransform(values, h)
            warped_hexagon.append(pointsOut.tolist()[0][0])
        warped_hexagons.append(warped_hexagon)

    return warped_hexagons


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

    # example_data = example_data[0:10000]

    src = [
        [1124, 1416],
        [1773, 2470],
        [3785, 1267],
        [3416, 928],
        [2856, 1303],
        [2452, 916],
        [3672, 2476]
    ]
    dst = [
        [50.971296, 11.037630],
        [50.971173, 11.037914],
        [50.971456, 11.037915],
        [50.971705, 11.037711],
        [50.971402, 11.037796],
        [50.971636, 11.037486],
        [50.971214, 11.038025]
    ]

    h, _ = calculateHomographyMatrix(src, dst)
    latlon_coordinates = transformPoints(example_data, h)

    hexbin = Hexbin(0.5)
    hexbin.add_data(latlon_coordinates)
    latlon_hexagons = hexbin.hexagon_points()
    
    hexagon_fills = [x[1] for x in hexbin.hexagons().items()]
    scale = Colorscale([0, 40])
    # domain = hexbin._get_minmax()
    for i in range(len(hexagon_fills)):
        hexagon_fills[i] = scale.get_color(hexagon_fills[i][4])
        # print(hexagon_fills[i])
        # exit()

        # hexagon_fills[i] = 1

    _, h_inverse = cv2.invert(h)
    warped_hexagons = transformHexagons(latlon_hexagons, h_inverse)

    writer = Writer("text.svg", image="bg.jpg", dimensions=(5952, 3348))
    writer.add_hexagons(warped_hexagons, hexagon_fills)
    writer.add_circles(example_data, radius=1)
    writer.save()


