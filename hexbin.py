from math import sqrt, pi, sin, cos
import svgwrite

class Hexbin():

    radius = 1
    bins = {}

    def __init__(self, data, radius):

        self.radius = radius
        self.w = sqrt(3) * self.radius
        self.h = 2 * self.radius
        self.dx = self.w
        self.dy = 0.75 * self.h

        for i in range(0, len(data)):

            px = data[i][0]
            py = data[i][1]

            # j rows
            # i columns

            pj = round(py / self.dy)
            pi = round(px / self.dx - (pj & 1) / 2)

            pj = int(pj)

            bin_id = "{}-{}".format(pi, pj)
            try:
                self.bins[bin_id][4] += 1
            except KeyError as e:
                self.bins[bin_id] = (px, py, pi, pj, 1)

            print("{}, {} : {} {}".format(px, py, pi, pj))


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


    def save_svg(self, filename):

        if not filename.endswith(".svg"):
            filename += ".svg"

        dwg = svgwrite.Drawing(filename, profile='tiny', size = ("800px", "600px"))
        dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
        dwg.add(dwg.text('Test', insert=(0, 0.2), fill='red'))

        for key, item in self.bins.items():
            dwg.add(dwg.path(self._get_hexagon_path(self.radius), transform="translate({},{})".format(self.dx*item[2], self.dy*item[3])))

        dwg.save()


if __name__ == "__main__":

    example_data = [
        [1, 1],
        [50, 2],
        [3, 70],
        [80, 75]
    ]

    hexbin = Hexbin(example_data, 10)
    hexbin.save_svg("text.svg")
