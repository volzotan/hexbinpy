# rsvg-convert -h 5952 text.svg > text.png

CURDIR=`/bin/pwd`

/Applications/Inkscape.app/Contents/Resources/bin/inkscape --export-png $CURDIR"/output.png" $CURDIR"/output.svg"