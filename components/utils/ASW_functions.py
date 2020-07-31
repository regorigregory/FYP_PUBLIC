import numpy as np


#Eucledian distance of color intensities

def get_color_distance(p,q):

    return np.sqrt(np.square(np.subtract(p,q)))

#Eucledian distance of pixel coordinates

def get_spacial_distance_weights(window):
    spatial_distance = np.zeros(window.shape)
    p_x = int(window.shape[0] / 2)
    p_y = int(window.shape[1] / 2)
    center_coords = (p_y, p_x)
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            d_y = np.square(center_coords[0]-i)
            d_x = np.square(center_coords[1]-j)
            spatial_distance[i, j] = np.sqrt(d_y+d_x)
    # as the spatial difference for one window size will always
    #be the same throughout the image
    return spatial_distance

def get_color_rule_component(window, gamma_c):
    p_x = int(window.shape[0]/2)
    p_y = int(window.shape[1]/2)
    reference_pixel = window[p_y, p_x]
    delta_c = get_color_distance(reference_pixel, window)
    return np.exp(-(delta_c/gamma_c))

def get_spatial_rule_component(gamma_s, spacial_distance_weight):
    delta_s = spacial_distance_weight
    return np.exp(-(delta_s/gamma_s))

def get_bilateral_suport_weights(window, gamma_c, gamma_s):
    w_c = get_color_rule_component(window, gamma_c)
    spatial_weights = get_spacial_distance_weights(window)
    w_s = get_spatial_rule_component(gamma_s, spatial_weights)
    return w_c*w_s


def get_boundary_strength_weights(window):
    pass

def get_boundary_rule_component(window, gamma_b):
    pass
    delta_b = get_boundary_strength_weights()

    return np.exp(-(delta_b / gamma_b))





def get_boundary_strength():
    pass

if __name__ == "__main__":
    window = np.random.randint(0, 255, [5,5])
    bilateral_suport_weights = get_bilateral_suport_weights(window, 10, 10)
    pass
