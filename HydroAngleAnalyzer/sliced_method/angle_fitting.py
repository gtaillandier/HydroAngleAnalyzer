#angle fit

def circle_equation(XY, Xs, Ys, R):
    """
    Equation of a circle for fitting.
    
    Parameters:
    XY : tuple of arrays
        Coordinates (X, Y).
    Xs : float
        X-coordinate of circle center.
    Ys : float
        Y-coordinate of circle center.
    R : float
        Radius of the circle.

    Returns:
    array-like
        Difference from the circle equation.
    """
    X, Y = XY
    return np.sqrt((X - Xs)**2 + (Y - Ys)**2) - R



def contact_angle(o_coords,delta_gamma, max_dist, o_center_geom, z_wall , y_width, delta_y_axis):
    gamma =0
    y_axis_list = np.arange(0, y_width, delta_y_axis)

    nn = 100
    delta_angle = 4
    limit_med = 9.5
    list_alfas = []
    array_surfaces = []
    array_popt = []
    for y_axis_value in y_axis_list:
        diff_cont = []
        # Calculate interface positions and XZ coordinates
        #print(f"gamma: {gamma}")
        o_center_geom[1] = y_axis_value
        list_rr, list_xz = multiple_line(o_coords, delta_angle, nn, max_dist, gamma=gamma, o_center_geom= o_center_geom)
        surf = np.array(list_xz)
        array_surfaces.append(surf)
        list_rr = np.array(list_rr)
        #save_array_as_txt(surf, f'allsurf_angle_{gamma}.txt')
        listmed =[limit_med]

            # Separate the surface data into two groups based on a threshold    
        surf_line = surf[(surf[:, 1] > limit_med )]
        #save_array_as_txt(surf_line, f'surf_line_angle_{gamma}.txt')
            # Prepare data for fitting the circle
        X_data = surf_line[:, 0]
        Y_data = surf_line[:, 1]
        mean_rr = np.mean(list_rr[:, 0])
            # Initial guess for the circle parameters
        initial_guess = [o_center_geom[0], o_center_geom[2], mean_rr]
        #print('center geom init :')
        #print(o_center_geom)
        bound = [(-max_dist-o_center_geom[0], o_center_geom[0]+max_dist) , (-max_dist+o_center_geom[2],max_dist+o_center_geom[2]) , (0 ,10+ mean_rr,)]
        #initial_guess = [0, 0, mean_rr, -9]
        #print("boundadries : ", bound)
        #bound = [(-max_dist, max_dist) , (-max_dist,max_dist) , (-max_dist - mean_rr ,max_dist + mean_rr), (-15 ,-8) ]

        lower_bounds, upper_bounds = zip(*bound)
        # Fit the circle equation to the data
        popt, _ = curve_fit(circle_equation, (X_data, Y_data), np.zeros_like(X_data), p0=initial_guess,bounds=(lower_bounds, upper_bounds))
        array_popt.append(popt)
        Xs, Ys, R = popt
        #print('pOpt:')
        #print(popt) 
            # Find the intersection points with y = 60
        #print("limit med: " ,limit_med)
        y_line = z_wall
        delta_y = y_line - Ys
        discriminant = R**2 - delta_y**2
        if discriminant < 0:
            #print("No intersection points ")
            diff_cont.append(None)
                #raise ValueError("No intersection points found with y = 60")

        else :# Calculate x-intersections
            x_intersections = Xs + np.array([-1, 1]) * np.sqrt(discriminant)
            x_intersection = x_intersections[0]

            # Calculate the tangent line at the first intersection point
            dx = x_intersection - Xs
            dy = y_line - Ys
            tangent_slope = -dx / dy
            tangent_angle = np.arctan(tangent_slope)
            diff_cont.append(np.degrees(tangent_angle))

            # Output the contact angle
        #print(diff_cont)
        if diff_cont[0] != None:
            list_alfas.append(np.abs(diff_cont[0]))
    return list_alfas, array_surfaces, array_popt
