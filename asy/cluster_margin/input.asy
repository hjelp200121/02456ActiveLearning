settings.outformat = "pdf";

import poissonDisk;

unitsize(1.5cm, 1.5cm);
srand(12345);

path shape = scale(3.5) * shift(-0.5, -0.5) * unitsquare;



pair[] data = poissonDisk(3.5, 3.5, 0.25, 30, shape);
Bounds dataBounds = Bounds(data);

data = shift(-dataBounds.center()) * data;

drawPoints(data);
