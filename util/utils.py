import numpy as np

def pixel_to_xyz(depth_img, pixel_x, pixel_y, focal_length, image_center):
    # Get depth value at the specified pixel
    depth_value = depth_img[pixel_y, pixel_x]
    depth_value = depth_value*0.001
    u = (pixel_x - image_center[0])
    v = (pixel_y - image_center[1])
    coeff = np.sqrt(1 + ((u**2 + v**2)/(focal_length**2)))
    z = depth_value/coeff
    # Convert pixel coordinates to NDC (normalized device coordinates)
    ndc_x = (pixel_x - image_center[0]) / focal_length
    ndc_y = (pixel_y - image_center[1]) / focal_length

    # Calculate XYZ coordinates in camera space
    # Assuming the camera's optical axis is the Z-axis and the image plane is at Z = focal_length
    # X and Y coordinates are proportional to NDC coordinates
    x = z * ndc_x
    y = z * ndc_y
    # z = depth_value

    return x, y, z