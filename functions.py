def line_fit(points):
    coefficients=[0,0,0]
    no_points = len(points)
    sum_y =0.
    sum_x = 0.
    sum_xy = 0.
    sum_x2 = 0.
    sum_x2y =0.
    sum_x3 = 0.
    sum_x4 = 0.
    S_xx = 0.
    S_xy = 0.
    S_xx2 = 0. 
    S_x2y = 0.
    S_x2x2 = 0. 
    avg_y = 0. 
    avg_x = 0. 

    for i in range(no_points):
        x2 = points[i].y * points[i].y
        sum_y   += points[i].x
        sum_x   += points[i].y
        sum_xy  += points[i].y * points[i].x
        sum_x2  += x2
        sum_x2y += x2 * points[i].x
        sum_x3  += points[i].y * x2
        sum_x4  += x2 * x2
    
    avg_y = sum_y / no_points
    avg_x = sum_x / no_points

    S_xx   = sum_x2  - sum_x  * avg_x
    S_xy   = sum_xy  - sum_x  * avg_y
    S_xx2  = sum_x3  - sum_x2 * avg_x
    S_x2y  = sum_x2y - sum_x2 * avg_y
    S_x2x2 = sum_x4  - sum_x2 * sum_x2 / no_points
    scaler = 1. / (S_xx* S_x2x2 - S_xx2*S_xx2)

    coefficients[2] = (S_x2y * S_xx - S_xy * S_xx2) * scaler
    coefficients[1] = (S_xy * S_x2x2 - S_x2y * S_xx2) * scaler
    coefficients[0] = (sum_y - coefficients[1] * sum_x - coefficients[2] * sum_x2) / no_points

    return coefficients

def ransac_inl(points,coefficients,threshold):
    inliers = 0
    for point in points:
        dist = abs(coefficients[2]*point.y*point.y+coefficients[1]*point.y+coefficients[0]-point.x)
        if dist<threshold:
            inliers=inliers+1
    return inliers