from hexbin import Hexbin, Colorscale, Alphascale

import time
import csv
import svgwrite
import json
import os

import numpy as np
import cv2

# INPUT_DIR           = "/Users/volzotan/Documents/DESPATDATASETS/19-11-24_herderplatz_aligned_annotation"
# DIMENSIONS          = (4000, 3000)
# BACKGROUND_FILE     = "herderplatz2.jpg"
# HEXAGON_SIZE        = 20 #30 # 40 #px

INPUT_DIR           = "/Users/volzotan/Documents/DESPATDATASETS/19-12-21_augustbaudertplatz_annotation/blobdetector_history_500"
BACKGROUND_FILE     = "augustbaudertplatz2.jpg"
HEXAGON_SIZE        = 20

# INPUT_DIR           = "/Users/volzotan/Documents/DESPATDATASETS/19-12-20_frauenplan_aligned_annotation"
# DIMENSIONS          = (2720, 1530)
# BACKGROUND_FILE     = "frauenplan.jpg"
# HEXAGON_SIZE        = 20 

# ----------------------------------------

# [FORWARD_PIXEL -> FORWARD_GEO] -> [BACKWARD_GEO -> BACKWARD_PIXEL]
# data acquisition images           nice display beautyshot image(s)

# AUGUST BAUDERT PLATZ

MAPPING_FORWARD_PIXEL = [
    [24, 203],
    [97, 1457],
    [2573, 1474],
    [2692, 259],
    [1131, 737]
]

MAPPING_FORWARD_GEO = [
    [50.991123, 11.325811],
    [50.990763, 11.325773],
    [50.990727, 11.326918],
    [50.991077, 11.326979],
    [50.990958, 11.326268]
]


MAPPING_BACKWARD_GEO = [
    [50.991122, 11.325214],
    [50.990562, 11.326270],
    [50.990557, 11.326506],
    [50.990930, 11.327131],
    [50.991074, 11.327012]
]

MAPPING_BACKWARD_PIXEL = [
    [932, 336],
    [268, 2312],
    [576, 2852],
    [3660, 2876],
    [3624, 2116]
]

# ----------------------------------------

CSS_FILENAME        = "style.css"

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
            
            fill_value = None
            if len(fills[i] == 4): # contains alpha
                fill_value = svgwrite.rgba(fills[i][0]*255, fills[i][1]*255, fills[i][2]*255, fills[i][3])
            else:
                fill_value = svgwrite.rgb(fills[i][0]*255, fills[i][1]*255, fills[i][2]*255)

            self.dwg.add(self.dwg.path(Hexbin.create_svg_path(hexagons[i], absolute=True), fill=fill_value)) 

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
                [fills[i][0]*255, fills[i][1]*255, fills[i][2]*255, fills[i][3]]
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
                # out.write("\" fill=\"rgba({},{},{},{})\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), int(h[1][3])))
                out.write("\" fill=\"rgb({},{},{})\" fill-opacity=\"{}\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), h[1][3]))
 
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

    time_start = time.time()

    detections = []
    raw_bounding_boxes = []

    skip = 0

    json_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append((root, f))

    json_files = sorted(json_files, key=lambda item: item[1])

    print("found {} json files".format(len(json_files)))

    for f in json_files:
        with open(os.path.join(f[0], f[1])) as json_file:
            data = json.load(json_file)

            for box in data["boxes"]:
                minx = float(box[0])
                miny = float(box[1])
                maxx = float(box[2])
                maxy = float(box[3])

                detections.append([minx + (maxx-minx)/2.0, maxy])
        
    print("loaded {} boxes, filtered {}".format(len(detections), skip))

    # NON HOMOGRAPHY

    # hexbin = Hexbin(HEXAGON_SIZE, assume_diameter_in_metres=False)
    # hexbin.add_data(detections)
    # hexagons = hexbin.hexagon_points()

    # HOMOGRAPHY

    h1, _ = calculateHomographyMatrix(MAPPING_FORWARD_PIXEL, MAPPING_FORWARD_GEO)
    h2, _ = calculateHomographyMatrix(MAPPING_BACKWARD_GEO, MAPPING_BACKWARD_PIXEL)

    latlon_coordinates = transformPoints(detections, h1)

    hexbin = Hexbin(0.5)
    hexbin.add_data(latlon_coordinates) 
    latlon_hexagons = hexbin.hexagon_points()

    hexbin.calculate_neighbour_values()
    hexagon_fills = [x[1] for x in hexbin.hexagons().items()]

    ## hexagon values (color scale)
    ## fill hexagons with viridis colormap based on their own value
    hexagon_binsizes = [x[4] for x in hexagon_fills]
    hexagon_binsizes = sorted(hexagon_binsizes)
    maxval = hexagon_binsizes[int(len(hexagon_binsizes)*0.99)]
    print("maximum hexagon value: {}".format(maxval))
    scale = Colorscale([0, maxval])

    ## hexagon neighbour values (alpha scale)
    ## decrease opacity when empty neighbouring cells are in the vicinity
    # hexagon_binsizes = [x[4] for x in hexagon_fills]
    # hexagon_binsizes = sorted(hexagon_binsizes)
    # maxval = hexagon_binsizes[int(len(hexagon_binsizes)*0.99)]
    # print("maximum hexagon value: {}".format(maxval))
    # scale = Alphascale([0, maxval])

    # hexagon neighbour values (alpha scale)
    # 
    hexagon_binsizes = [x[5] for x in hexagon_fills]
    hexagon_binsizes = sorted(hexagon_binsizes)
    maxval = hexagon_binsizes[int(len(hexagon_binsizes)*0.99)]
    print("maximum hexagon neighbour value: {}".format(maxval))
    ascale = Alphascale([0, maxval/8.0])

    # print(*hexagon_fills, sep="\n")

    for i in range(len(hexagon_fills)):    
        # hexagon_fills[i] = (*scale.get_color(hexagon_fills[i][4]), ascale.get_alpha(hexagon_fills[i][4])) # merge color tuple and alpha value (alpha by hex val)
        hexagon_fills[i] = (*scale.get_color(hexagon_fills[i][4]), ascale.get_alpha(hexagon_fills[i][5])) # merge color tuple and alpha value (alpha by neighbour val)
        # hexagon_fills[i] = (*scale.get_color(hexagon_fills[i][4]), 1.0)                                   # no alpha channel, only color
        # hexagon_fills[i] = (0, 0, 1, scale.get_alpha(hexagon_fills[i][4]))                                # no color, only alpha channel
        
    # NON HOMOGRAPHY

    # warped_hexagons = hexagons

    # HEXAGON HOMOGRAPHY CALCULATIONS

    # _, h_inverse = cv2.invert(h)
    # _, h_inverse = cv2.invert(h)
    warped_hexagons = transformHexagons(latlon_hexagons, h2)

    print("start saving. elapsed time: {0:.2f} seconds".format(time.time()-time_start))

    background_image = cv2.imread(BACKGROUND_FILE)
    image_dimensions = background_image.shape
    image_dimensions = [image_dimensions[1], image_dimensions[0]]

    writer = WriterSimple("output.svg", dimensions=image_dimensions, image=BACKGROUND_FILE)
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