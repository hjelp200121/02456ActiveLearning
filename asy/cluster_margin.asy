settings.outformat = "pdf";


size(10cm, 6cm);
srand(1234);

real pi = radians(180);

pair[] poissonDisk(real width, real height, real minDist, int quality, bool retain(pair)) {


    real cellSize = minDist / sqrt(2.0);

    int l1 = (int) ceil(width / cellSize);
    int l2 = (int) ceil(height / cellSize);


    pair[] grid = array(l1 * l2, 2.0 * (-minDist, -minDist)); // points with negative coordinates always outside
    pair[] points;    
    pair[] queue;

    void gridPut(pair p) {
        int x = (int) ((p.x / width) * l1);
        int y = (int) ((p.y / height) * l2);
        grid[x + l1 * y] = p;
    }

    bool gridReject(pair p) {
        
        int x = (int) ((p.x / width) * l1);
        int y = (int) ((p.y / height) * l2);

        for (int y_ = y - 2; y_ < y + 3; y_ += 1) {
            for (int x_ = x - 2; x_ < x + 3; x_ += 1) {
                
                if (x_ < 0 || x_ >= l1 || y_ < 0 || y_ >= l2) {
                    continue;
                }

                pair q = grid[x_ + l1 * y_];

                if (length(q - p) < minDist) {
                    return true;
                }
            }
        }

        return false;
    }

    pair randPopQueue() {
        int i = (int) (queue.length * unitrand());
        pair p = queue[i];
        queue.delete(i);

        return p;
    }

    pair sampleAnnulus(pair c) {
        real angle = 2.0 * pi * unitrand();
        real radius = minDist * (unitrand() + 1.0);

        return c + radius * (cos(angle), sin(angle));
    }

    
    pair first = (width * unitrand(), height * unitrand());
    gridPut(first);

    points.push(first);
    queue.push(first);

    while (queue.length != 0) {

        pair p = randPopQueue();

        for (int i = 0; i < quality; i += 1) {
            
            pair q = sampleAnnulus(p);
            
            if (q.x < 0.0 || q.x >= width || q.y < 0.0 || q.y >= height) {
                continue;
            }
            
            if (!retain(q)) {
                continue;
            }

            if (gridReject(q)) {
                continue;
            }

            points.push(q);
            queue.push(q);
            gridPut(q);
        }
    }
    
    // for(int y = 0; y < l2; y += 1) {
    //     for(int x = 0; x < l1; x += 1) {

    //         pen c;

    //         if (grid[x + l1 * y].x < 0.0) {
    //             c = green;
    //         } else {
    //             c = red;
    //         }
            
    //         dot((x, y) * cellSize, c);
    //     }
    // }

    return points;
}

bool isInCircle(pair p) {
    return length(p - (1.0, 1.0)) < 1.0;
}

pair[] filterPoints(pair[] points) {
    
    pair[] filtered;

    for(int i = 0; i < points.length; i += 1) {
        pair p = points[i];
        real retainProb = 0.5 + 0.5 * exp(-dot(p, p));

        if(unitrand() < retainProb) {
            filtered.push(p);
        }
    }

    return filtered;
}

void drawPoints(picture pic=currentpicture, pair[] points, pen p=black) {
    for(int i = 0; i < points.length; i += 1) {
        filldraw(pic, circle(points[i], 0.075), p+opacity(0.4), p);
    }
}

transform[] transforms = {
    shift(0.0, 0.0),
    shift(1.8, -0.9),
    shift(0.5, -2.5),
    shift(2.4, -3.4),
};

pair[][] clusters;

for(int i = 0; i < transforms.length; i += 1) {
    pair[] cluster = poissonDisk(2.0, 2.0, 0.2, 30, isInCircle);
    cluster = filterPoints(shift(-1, -1) * cluster);
    cluster = transforms[i] * cluster;
    clusters.push(cluster);
}

pair pmin = 1000 * (1, 1);
pair pmax = 1000 * (-1, -1);

for(int i = 0; i < clusters.length; i += 1) {

    pair[] cluster = clusters[i];

    for (int j = 0; j < cluster.length; j += 1) {
        pmin = (min(pmin.x, cluster[j].x), min(pmin.y, cluster[j].y));
        pmax = (max(pmax.x, cluster[j].x), max(pmax.y, cluster[j].y));
    }
}


void pic1() {
    transform t = shift(0.0, 0.0);

    for(int i = 0; i < clusters.length; i += 1) {
        drawPoints(t * clusters[i]);
    }

    draw(t * (pmin--(pmax.x, pmin.y)--pmax--(pmin.x, pmax.y)--cycle));
}

void pic2() {
    transform t = shift(0.0, 0.0);

    for(int i = 0; i < clusters.length; i += 1) {
        drawPoints(t * clusters[i]);
    }

    draw(t * (pmin--(pmax.x, pmin.y)--pmax--(pmin.x, pmax.y)--cycle));
}

pic1();

draw((0, 0)--(1, 0));
draw((0, 0)--(0, 1));

