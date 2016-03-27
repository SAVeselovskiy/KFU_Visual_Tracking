xx = 1
yy = 1
matr = [[1,2,3],[5,6,7],[9,10,11],[13,14,15]]
for row in matr:
    print row
is_end = False
layer = 1
while not is_end:
    is_end = True
    for start_point, vector in (([-1,-1],[1,0]),([1,-1],[0,1]),([1,1],[-1,0]),([-1,1],[0,-1])):
        x = xx + start_point[0]*layer + vector[0]
        y = yy + start_point[1]*layer + vector[1]
        while x < len(matr) and x >= 0 and y < len(matr[x]) and y >= 0 and xx - layer <= x <= xx + layer and yy - layer <= y <= yy + layer:
            is_end = False
            print x, y, matr[x][y]
            x += vector[0]
            y += vector[1]
    layer += 1

