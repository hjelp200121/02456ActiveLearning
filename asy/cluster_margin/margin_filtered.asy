settings.outformat = "pdf";

import poissonDisk;

unitsize(1.5cm, 1.5cm);
srand(12345);

pair[] filterPoints(pair[] points) {
    
    pair[] filtered;

    for(int i = 0; i < points.length; i += 1) {
        if (unitrand() > 0.7) {
            filtered.push(points[i]);
        }
    }

    return filtered;
}

path[] shapes = {
    rotate(55) * ellipse((0, 0), 0.9, 0.8),
    rotate(-32) * ellipse((0, 0), 1.3, 0.6),
    rotate(71) * ellipse((0, 0), 1.1, 0.9),
    rotate(8) * ellipse((0, 0), 1.2, 0.7),
};

transform[] transforms = {
    shift(0.0, 0.0),
    shift(2.0, -0.9),
    shift(0.3, -2.0),
    shift(2.3, -2.5),
};

pair[][] clusters;

for(int i = 0; i < transforms.length; i += 1) {
    pair[] cluster = poissonDisk(3.0, 3.0, 0.2, 5, shapes[i]);
    cluster = transforms[i] * cluster;
    cluster = filterPoints(cluster);
    clusters.push(cluster);
}

for(int i = 0; i < clusters.length; i += 1) {
    drawPoints(clusters[i]);
}

for(int i = 0; i < clusters.length; i += 1) {
    draw(transforms[i] * scale(1.05) * shapes[i]);
}


