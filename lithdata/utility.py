import numpy as np
from lithdata.constants import kB
import pickle

def concatenate_add_center(listA, listB):
    """With letters as cartesian points [x, y]:
    [a, b, c] + [d, e, f] = [a, b, c + d, e f]
    """
    return np.vstack((listA[:-1], listA[-1:] + listB[:1], listB[1:]))


def signed_area_np(x, y):
    """Calculates the signed area of a polygon.

    From https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon/717367#717367
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    n = len(x)
    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)    
    return (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0

def find_nearest(array, value): #
    """Finds the nearest value in an array to the given value."""
    if value == np.inf:
        idx = array.argmax()
    elif value == -np.inf:
        idx = array.argmin()
    else:
        idx = (np.abs(array - value)).argmin()
    return array[idx]

def pickle_obj(obj, filename):
    output = open(filename, 'wb')
    pickle.dump(obj, output)
    output.close()

#Creates (endTimeStep-startTimeStep)/stepSize arrays that are of the same form as the ones created by the createGrid function
#Useful for creating my own averaged quantities that SPARTA doesn't calculate for me
def parse_timesteps(file, start_timestep, end_timestep, flag_text="ITEM: CELL"):
    n_cells = None
    all_values = []
    with open(file, 'rt') as f:
        for line in f:
            if "ITEM: TIMESTEP\n" == line:
                line = f.readline()
                timestep = int(line)
                if timestep > end_timestep: break
                if timestep >= start_timestep and n_cells is None: # initialize totals array
                    _ = f.readline()
                    n_cells = int(f.readline())
            elif flag_text in line and n_cells is not None:
                values = []
                for _ in range(n_cells):
                    line = f.readline()
                    values.append([float(i) for i in line.split()])
                all_values.append(np.array(values))
    return np.array(all_values)

def parse_timestep(file, timestep):
    x = parse_timesteps(file, timestep, timestep)
    return np.squeeze(x, axis=0)

def parse_grid(file, timestep=0):
    """Read a file that defines the SPARTA grid."""
    return parse_timestep(file, timestep)

# ---------------------------------
# physics formulas

def langmuir_flux(density, t_kelvin, mass):
    """Calculates langmuir flux of particles # / m^2 s
    density in #/m^3
    t_kelvin is temperature in kelvin
    mass in kg    
    """
    flux = density * np.sqrt(kB * t_kelvin / (2 * np.pi * mass))
    return flux

# ---------------------------------
# code to interpret SPARTA collision codes

def cylindrical_ring_area(coords):
    """Calculate the area of a ring element.
    coords is either a list
    [z1, r1, z2, r2]
    or an array 
    [[z1,], [r1,], [z2,],[r2,]]"""
    z1, r1, z2, r2 = coords
    r_center = np.average((r1, r2), axis=0)
    delta_z = z2 - z1
    delta_r = r2 - r1
    return 2 * np.pi * r_center * np.hypot(delta_z, delta_r)

def line_lengths(coords):
    """Calculate the area of a ring element.
    coords is either a list
    [x1, y1, x2, y2]
    or an array 
    [[x1,], [y1,], [x2,],[y2,]]"""
    x1, y1, x2, y2 = coords
    delta_y = y2 - y1
    delta_x = x2 - x1
    return np.hypot(delta_x, delta_y)

def wall_segment_centers(collision_array):
    """Takes an array with shape (6, N) representing N segments.
    The rows represent [?, ?, x1, y1, x2, y2].    
    """
    x_centers = (collision_array[2] + collision_array[4]) / 2
    y_centers = (collision_array[3] + collision_array[5]) / 2
    return [x_centers, y_centers]

#--------------------------------

def areas_normal_to_x(grid_cells, cutoff_radius=None, axisymmetric=True):
    """Areas of the sides of cells normal to the symmetry axis.
    
    axisymmetric: False for 2D cartesian
    
    Expects rows of grid cells like
    
    [id xlo ylo xc yc xhi yhi vol].
    Computes pi * (yhi^2 - ylo^2).
    """
    index_of_max_radius = np.argmax(grid_cells[:, 6])
    areas = np.pi * (grid_cells[:, 6]**2 - grid_cells[:, 2]**2)
    max_yhi = grid_cells[index_of_max_radius, 6]
    if cutoff_radius is not None and max_yhi > cutoff_radius:
        areas[index_of_max_radius] -= np.pi * (max_yhi**2 - cutoff_radius**2)
    if not axisymmetric:
        areas = grid_cells[:, 6] - grid_cells[:, 2]
        if cutoff_radius is not None and max_yhi > cutoff_radius:
            areas[index_of_max_radius] -= (max_yhi - cutoff_radius)
    return areas

def extract_data_at_xc(xc, grid_cells, data, max_radius=None):
    """returns data and grid information at a selected xc"""
    x_centers = grid_cells[:, 3]
    selector = x_centers == xc
    if max_radius is not None:
        y_lo = grid_cells[:, 2]
        selector = np.logical_and(selector, y_lo < max_radius)
    good_data = data[selector]
    good_cells = grid_cells[selector]
    return {'data': good_data, 'cells': good_cells}

def compute_flux_past(xc, vbs, max_radius=None, species=0):
    grid = vbs.grid_cells
    velocity = np.nan_to_num(vbs.grid_values[:,vbs.values_column_i['vx_' + str(species)]])
    density  = np.nan_to_num(vbs.grid_values[:,vbs.values_column_i['n_' + str(species)]])
    velocities = extract_data_at_xc(xc, grid, velocity, max_radius=max_radius)
    densities = extract_data_at_xc(xc, grid, density, max_radius=max_radius)['data']
    good_cells = velocities['cells']
    velocities = velocities['data']
    cell_flux_areas = areas_normal_to_x(good_cells, cutoff_radius=max_radius)
    return np.sum(densities * velocities * cell_flux_areas)
