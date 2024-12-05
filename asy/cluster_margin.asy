settings.outformat = "pdf";


size(10cm, 6cm);
srand(1234);

real pi = radians(180);

pair[] poissonDisk(real width, real height, real minDist, int quality) {


    real cellSize = minDist / sqrt(2.0);

    int l1 = (int) ceil(width / cellSize);
    int l2 = (int) ceil(height / cellSize);
    pair[] grid = array(l1 * l2, 2.0 * (-minDist, -minDist)); // points with negative coordinates always outside
    pair[] points;    
    pair[] queue;

    void gridPut(pair p) {
        int x = (int) (p.x * l1);
        int y = (int) (p.y * l2);
        grid[x + l1 * y] = p;
    }

    bool gridReject(pair p) {
        
        int x = (int) (p.x * l1);
        int y = (int) (p.y * l2);

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

            if (gridReject(q)) {
                continue;
            }

            points.push(q);
            queue.push(q);
            gridPut(q);
        }
    }

    return points;
}

pair[] jitter(real width, real height, int l1, int l2) {

    pair[] grid = array(1000, l1 * l2);
}

pair[] points = poissonDisk(1.0, 1.0, 0.05, 30);

transform t1 = shift((0, 0));
transform t2 = shift((2, 0));

for(int i = 0; i < points.length; i += 1) {
    dot(t1 * points[i]);
}

for(int i = 0; i < points.length; i += 1) {
    dot(t2 * points[i]);
}

