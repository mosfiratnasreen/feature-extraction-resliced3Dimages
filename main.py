import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, sobel

# DATA LOADING


def load_data():  # loads image and spacing from npy files
    if os.path.exists('image.npy') and os.path.exists('spacing.npy'): #check for files first
        print(f"loading data from npy files")
        volume = np.load('image.npy')
        spacing = np.load('spacing.npy')
    else:
        #fallback to simpleITK
        try:
            import SimpleITK as sitk
            print(f"npy files not found - loading .gipl")
            img = sitk.ReadImage('case001_trus.gipl')
            volume = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = np.array(img.GetSpacing()[::-1])
        except (ImportError, RuntimeError):
            print("error could not load data")
            sys.exit(1)
    print(f"volume shape {volume.shape}")
    print(f"voxel spacing {spacing}")
    return volume, spacing


# RESLICING
def get_reslice_coordinates(center, normal, spacing, output_shape, pixel_size=0.5): #generates 3D sampling coordinates for 2D plane
    #center (x,y,z) in volume where slice centre should be
    #normal vector (dx, dy, dz) perpendicular to the plane
    #spacing (sx, sy, sz) physical size of one voxel
    #pixel size how many mm each pixelshould be in new image
    h, w = output_shape #resolution of the output 2d image

    normal = np.array(normal, dtype=np.float64) #convert to np array
    normal /= np.linalg.norm(normal) #normalise the normal vector to 1

    #orthogonal (u,v) basis for plane
    if np.abs(normal[0]) < 0.9:
        arbitrary = np.array([1,0,0]) #arbitrary that is not parallel to the norm
    else:
        arbitrary = np.array([0,1,0])

    v_vec = np.cross(normal, arbitrary) #cross product will lie flat on plane
    v_vec /= np.linalg.norm(v_vec)

    u_vec = np.cross(normal, v_vec) #u_vec would be perpendicular to both
    u_vec /= np.linalg.norm(u_vec)

    #scale vectors by physical spacing  mm to indices
    step_u = u_vec * (pixel_size / spacing) #number of indices in 3D to move 1 pixel in 2D >
    step_v = v_vec * (pixel_size / spacing) # same to go 1 pixel DOWN in 2D

    #create 2d grid
    xs = np.arange(-w//2, w//2) #range of numbers centered around 0
    ys = np.arange(-h//2, h//2)
    grid_x, grid_y = np.meshgrid(xs, ys) #2d matrices

    #point = centre + (x * u_vec) + (y * v_vec)
    #reshape everything to (3,1)
    centre_col = np.array(center).reshape(3,1)
    step_u_col = step_u.reshape(3,1)
    step_v_col = step_v.reshape(3,1)

    flat_x = grid_x.flatten() #to 1D arrays to make the math easier
    flat_y = grid_y.flatten()

    coords = centre_col + (step_u_col * flat_x) + (step_v_col * flat_y) #(3,N) array
    return coords, (h, w)










#################################################################################################
if __name__ == "__main__":
    volume, spacing = load_data()

    print("testing reslicing")
    real_centre = np.array(volume.shape) // 2
    real_normal = (0.0, 1.0, 1.0)
    output_shape = (256, 256)
    print(f"generating coordinates at {real_centre} and real normal {real_normal}")
    coordinates, shape_out = get_reslice_coordinates(real_centre, real_normal, spacing, output_shape)
    print(coordinates)
    print(shape_out)
