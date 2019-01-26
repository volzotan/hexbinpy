from hexbin import Hexbin, Colorscale

import time
import csv
import svgwrite
import yaml

import numpy as np
import cv2

import data as config

CSS_FILENAME = "style.css"

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
            # dwg.add(dwg.path(self._get_hexagon_path(self.radius), transform="translate({},{})".format(item[0], item[1]), fill="rgb(255,0,0)"))
            self.dwg.add(self.dwg.path(
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                fill=svgwrite.rgb(fills[i][0]*255, fills[i][1]*255, fills[i][2]*255))
            ) 

            # , 
            #     fill=svgwrite.rgb(0, 0, 0),
            #     stroke=svgwrite.rgb(0, 0, 0)
            # )

             # fill=svgwrite.rgb(*self._interpolate_color(item[4], self._get_minmax(), ((10, 10, 10), (200, 200, 200)))), 

    def add_circles(self, points, radius=3, fill=svgwrite.rgb(255, 0, 0)):
        for item in points:
            self.dwg.add(self.dwg.circle(center=item, r=radius, fill=fill))


    def save(self):
        self.dwg.save()


class WriterSimple():

    def __init__(self, filename, dimensions=None, image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename = filename
        self.dimensions = dimensions
        self.image = image

        self.hexagons = []
        self.circles = []
        self.rectangles = []

    def add_hexagons(self, hexagons, fills):
        for i in range(0, len(hexagons)):
            self.hexagons.append([
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                [fills[i][0]*255, fills[i][1]*255, fills[i][2]*255]
            ]) 

    def add_circles(self, circles, radius=3, fill=[255, 0, 0]):
        for item in circles:
            self.circles.append([item, radius, fill])


    def add_rectangles(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0):
        for item in coords:
            self.rectangles.append([item, strokewidth, stroke, opacity])


    def save(self):
        with open(self.filename, "w") as out:

            out.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?>")
            out.write("<?xml-stylesheet href=\"style.css\" type=\"text/css\" title=\"main_stylesheet\" alternate=\"no\" media=\"screen\" ?>")
            if self.dimensions is not None:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" width=\"{}px\" height=\"{}px\" ".format(self.dimensions[0], self.dimensions[1]))
            else:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" ")
            out.write("xmlns=\"http://www.w3.org/2000/svg\" xmlns:ev=\"http://www.w3.org/2001/xml-events\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" >")
            out.write("<defs />")
            if self.image is not None:
                out.write("<image x=\"0\" y=\"0\" xlink:href=\"{}\" />".format(self.image))

            for h in self.hexagons:
                out.write("<path d=\"")
                for cmd in h[0]:
                    out.write(cmd[0])
                    if (len(cmd) > 1):
                        out.write(str(cmd[1]))
                        out.write(" ")
                        out.write(str(cmd[2]))
                        out.write(" ")
                out.write("\" fill=\"rgb({},{},{})\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2])))
 
            for c in self.circles:
                out.write("<circle cx=\"{}\" cy=\"{}\" fill=\"rgb({},{},{})\" r=\"{}\" />".format(c[0][0], c[0][1], c[2][0], c[2][1], c[2][2], c[1]))

            for r in self.rectangles:
                out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], r[1], *r[2], r[3]))

            out.write("</svg>")


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

    # with open("data.yaml", 'r') as stream:
    #     input_file = yaml.safe_load(stream)

    time_start = time.time()

    detections = []
    raw_bounding_boxes = []
    latlon_coordinates_from_csv = []
    with open(config.INPUT_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        data = [r for r in csv_reader]
        data.pop(0) # remove header
        
        for line in data:

            if line[2] is not "0":
                continue

            lat = float(line[4])
            lon = float(line[5])

            minx = float(line[6])
            miny = float(line[7])
            maxx = float(line[8])
            maxy = float(line[9])

            latlon_coordinates_from_csv.append([lat, lon])
            detections.append([minx + (maxx-minx)/2, maxy])
            raw_bounding_boxes.append([minx, miny, maxx-minx, maxy-miny])

    # example_data = example_data[0:10000]

    print("loaded {} boxes".format(len(detections)))

    h, _ = calculateHomographyMatrix(config.src, config.dst)
    latlon_coordinates = transformPoints(detections, h)

    hexbin = Hexbin(0.5)
    hexbin.add_data(latlon_coordinates) #latlon_coordinates_from_csv)
    latlon_hexagons = hexbin.hexagon_points()
    
    hexagon_fills = [x[1] for x in hexbin.hexagons().items()]

    hexagon_binsizes = [x[4] for x in hexagon_fills]
    hexagon_binsizes = sorted(hexagon_binsizes)
    maxval = hexagon_binsizes[int(len(hexagon_binsizes)*0.99)]
    print("maximum hexagon value: {}".format(maxval))
    scale = Colorscale([0, maxval])

    # print(*hexagon_fills, sep="\n")

    for i in range(len(hexagon_fills)):
        hexagon_fills[i] = scale.get_color(hexagon_fills[i][4])
        # print(hexagon_fills[i])
        # exit()

        # hexagon_fills[i] = 1

    _, h_inverse = cv2.invert(h)
    warped_hexagons = transformHexagons(latlon_hexagons, h_inverse)

    print("start saving. elapsed time: {0:.2f} seconds".format(time.time()-time_start))

    writer = WriterSimple("output.svg", dimensions=config.DIMENSIONS, image=config.BACKGROUND_FILE)
    writer.add_hexagons(warped_hexagons, hexagon_fills)
    # writer.add_circles(detections, radius=1)
    # writer.add_circles(src, radius=40, fill=[0, 0, 0])
    # writer.add_circles(src, radius=20, fill=[255, 255, 255])
    # writer.add_rectangles(raw_bounding_boxes, strokewidth=1, stroke=[253, 231, 37], opacity=0.2)
    writer.save()

    print("done. elapsed time: {0:.2f} seconds".format(time.time()-time_start))


    #450457  69   4  87
    #1f968b  31 150 139
    #fde725 253 231  37