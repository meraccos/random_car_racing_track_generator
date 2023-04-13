import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import CubicSpline
import random
import yaml
import os

def generate_racing_map(width, height, track_width, num_control_points, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    racing_map = np.zeros((height, width))

    # Generate random angles and sort them
    angles = np.random.uniform(0, 2 * np.pi, num_control_points)
    angles = np.sort(angles)

    # Calculate the minimum radius based on track width
    min_radius = track_width / 2

    # Generate control points using random angles and random radii
    center_x, center_y = width // 2, height // 2
    control_points_x = center_x + (min_radius + np.random.uniform(0, min(width, height) / 4, num_control_points)) * np.cos(angles)
    control_points_y = center_y + (min_radius + np.random.uniform(0, min(width, height) / 4, num_control_points)) * np.sin(angles)

    # Close the loop
    control_points_x = np.append(control_points_x, control_points_x[0])
    control_points_y = np.append(control_points_y, control_points_y[0])

    # Generate a smooth closed loop track using cubic spline interpolation
    t = np.linspace(0, 1, num_control_points + 1)
    spline_x = CubicSpline(t, control_points_x, bc_type='periodic')
    spline_y = CubicSpline(t, control_points_y, bc_type='periodic')

    t_extended = np.linspace(0, 1, width)
    x = spline_x(t_extended)
    y = spline_y(t_extended)
    
    racing_map = np.zeros((height, width))
    half_track_width = track_width // 2

    for i in range(width):
        for j in range(height):
            min_distance = np.min(np.sqrt((x - i)**2 + (y - j)**2))

            if half_track_width - 1 <= min_distance <= half_track_width + 1:
                racing_map[j, i] = 1

    return racing_map, spline_x, spline_y

def racing_map_to_csv_data(racing_map, spline_x, spline_y, width, num_control_points, track_width):
    t_extended = np.linspace(0, 1, width)
    t_diff = np.gradient(t_extended)
    x = spline_x(t_extended)
    y = spline_y(t_extended)

    s_m = np.cumsum(np.sqrt(np.gradient(x)**2 + np.gradient(y)**2))

    x_m = x
    y_m = y

    psi_rad = np.arctan2(np.gradient(y), np.gradient(x))

    curvature = np.gradient(np.arctan2(np.gradient(y), np.gradient(x))) / np.sqrt(np.gradient(x)**2 + np.gradient(y)**2)
    kappa_radpm = curvature * t_diff

    # Assuming constant velocity and acceleration
    vx_mps = np.full(width, 6.0)
    ax_mps2 = np.full(width, 0.5)

    return np.column_stack((s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2))

def save_csv(data, file_name):
    df = pd.DataFrame(data, columns=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])
    df.to_csv(file_name, index=False, header=True, float_format='%.7f')


def save_yaml(file_name, image, resolution, origin, negate, occupied_thresh, free_thresh):
    data = {
        'image': image,
        'resolution': resolution,
        'origin': origin,
        'negate': negate,
        'occupied_thresh': occupied_thresh,
        'free_thresh': free_thresh,
    }

    with open(file_name, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

def save_image(racing_map, file_name):
    img = Image.fromarray(np.uint8((1 - racing_map) * 255), 'L')
    img.save(file_name)
    
def create_incremental_folder(prefix='map_'):
    i = 1
    while True:
        folder_name = f'{prefix}{i:02d}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        i += 1

if __name__ == '__main__':
    width = 800
    height = 400
    track_width = 12
    num_control_points = 5

    folder_name = create_incremental_folder()

    racing_map, spline_x, spline_y = generate_racing_map(width, height, track_width, num_control_points)
    csv_data = racing_map_to_csv_data(racing_map, spline_x, spline_y, width, num_control_points, track_width)
    
    save_csv(csv_data, os.path.join(folder_name, 'racing_map.csv'))
    save_image(racing_map, os.path.join(folder_name, 'racing_map.png'))
    save_yaml(os.path.join(folder_name, 'racing_map.yaml'), 'racing_map.png', 0.0625, [-78.21853769831466, -44.37590462453829, 0.0], 0, 0.45, 0.196)