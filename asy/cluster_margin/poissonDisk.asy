
pair[] poissonDisk(real width, real height, real minDist, int quality, path area) {
    pair gridCenter = 0.5 * (width, height);
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

    do {
        pair first = (width * unitrand(), height * unitrand());
        
        if (inside(area, first - gridCenter)) {            
            gridPut(first);
            queue.push(first);
            points.push(first - gridCenter);
        }
    } while(queue.length == 0);
    

    while (queue.length != 0) {

        pair p = randPopQueue();

        for (int i = 0; i < quality; i += 1) {
            
            pair q = sampleAnnulus(p);
            
            if (q.x < 0.0 || q.x >= width || q.y < 0.0 || q.y >= height) {
                continue;
            }
            
            if (!inside(area, q - gridCenter)) {
                continue;
            }

            if (gridReject(q)) {
                continue;
            }

            gridPut(q);
            queue.push(q);
            points.push(q - gridCenter);
        }
    }

    return points;
}

struct Bounds {
    pair min;
    pair max;

    void operator init() {
        this.min = 100000 * (1, 1);
        this.max = 100000 * (-1, -1);
    }

    void operator init(pair[] points) {
        pair min = 100000 * (1, 1);
        pair max = 100000 * (-1, -1);

        for (int j = 0; j < points.length; j += 1) {
            min = (min(min.x, points[j].x), min(min.y, points[j].y));
            max = (max(max.x, points[j].x), max(max.y, points[j].y));
        }

        this.min = min;
        this.max = max;
    }

    void operator init(Bounds a, Bounds b) {
        this.min = (min(a.min.x, b.min.x), min(a.min.y, b.min.y));
        this.max = (max(a.max.x, b.max.x), max(a.max.y, b.max.y)); 
    }

    void addMargin(real margin) {
        this.min -= (margin, margin);
        this.max += (margin, margin);
    }

    pair center() {
        return 0.5 * (this.min + this.max);
    }

    path outline() {
        return this.min--(this.max.x, this.min.y)--this.max--(this.min.x, this.max.y)--cycle;
    }
}

void drawPoints(picture pic=currentpicture, pair[] points, pen p=black) {
    for(int i = 0; i < points.length; i += 1) {
        // filldraw(pic, circle(points[i], 0.075), p+opacity(0.4), p);
        fill(pic, circle(points[i], 0.075), p);
    }
}

