from math import sqrt, pi, sin, cos
import svgwrite

class Hexbin():

    radius = 1

    def __init__(self, data, radius):

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
            px = px / self.dx - (pj & 1) / 2.0
            pi = round(px)
            py1 = py - pj;

            if (abs(py1) * 3 > 1):
                px1 = px - pi
                pi2 = pi + (-1 if px < pi else 1) / 2.0
                pj2 = pj + (-1 if py < pj else 1)
                px2 = px - pi2
                py2 = py - pj2
                if (px1 * px1 + py1 * py1 > px2 * px2 + py2 * py2):
                    pi = pi2 + (1 if pj & 1 else -1) / 2.0
                    pj = pj2

            bin_id = "{}-{}".format(pi, pj)
            try:
                self.bins[bin_id][4] += 1
            except KeyError as e:
                binx = (pi + (pj & 1) / 2.0) * self.dx
                biny = pj * self.dy
                self.bins[bin_id] = [binx, biny, pi, pj, 1]


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


    def _get_hexagon_path(self, r):

        cmd = self._draw_hexagon(r)
        cmd[0] = [["m"] + cmd[0]]
        cmd = cmd[0] + list(map(lambda x: ["l"] + x, cmd[1:]))
        cmd.append(["z"])

        return cmd


    def hexagons(self):
        return self.bins


    def save_svg(self, filename):

        if not filename.endswith(".svg"):
            filename += ".svg"

        dwg = svgwrite.Drawing(filename, profile='tiny', size = ("800px", "600px"))
        dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))

        for key, item in self.bins.items():
            dwg.add(dwg.path(self._get_hexagon_path(self.radius), transform="translate({},{})".format(item[0], item[1]), fill="rgb(50,50,50)"))

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

    example_data = [
        [1, 2],
        [1, 12]
    ]

    import numpy as np 
    mu, sigma = 0, 0.3  
    n = 1000
    x = np.multiply(np.random.normal(mu, sigma, n), 400)
    y = np.multiply(np.random.normal(mu, sigma, n), 300)
    example_data = [None] * n
    for i in range(0, n):
        example_data[i] = [float(x[i]), float(y[i])]

    hexbin = Hexbin(example_data, 10)
    hexbin.save_svg("text.svg")

    for key, item in hexbin.hexagons().items():
        print(item)


