import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, sobel

# DATA LOADING
def load_data():  # loads image and spacing from npy files
    if os.path.exists('image.npy') and os.path.exists('spacing.npy'):  # check for files first
        print(f"loading data from npy files")
        volume = np.load('image.npy')
        spacing = np.load('spacing.npy')
    else:
        # fallback to simpleITK
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
# generates 3D sampling coordinates for 2D plane
def get_reslice_coordinates(center, normal, spacing, output_shape, pixel_size=0.5):
    # center (x,y,z) in volume where slice centre should be
    # normal vector (dx, dy, dz) perpendicular to the plane
    # spacing (sx, sy, sz) physical size of one voxel
    # pixel size how many mm each pixelshould be in new image
    h, w = output_shape  # resolution of the output 2d image

    normal = np.array(normal, dtype=np.float64)  # convert to np array
    normal /= np.linalg.norm(normal)  # normalise the normal vector to 1

    # orthogonal (u,v) basis for plane
    if np.abs(normal[0]) < 0.9:
        # arbitrary that is not parallel to the norm
        arbitrary = np.array([1, 0, 0])
    else:
        arbitrary = np.array([0, 1, 0])

    v_vec = np.cross(normal, arbitrary)  # cross product will lie flat on plane
    v_vec /= np.linalg.norm(v_vec)

    u_vec = np.cross(normal, v_vec)  # u_vec would be perpendicular to both
    u_vec /= np.linalg.norm(u_vec)

    # scale vectors by physical spacing  mm to indices
    # number of indices in 3D to move 1 pixel in 2D >
    step_u = u_vec * (pixel_size / spacing)
    step_v = v_vec * (pixel_size / spacing)  # same to go 1 pixel DOWN in 2D

    # create 2d grid
    xs = np.arange(-w//2, w//2)  # range of numbers centered around 0
    ys = np.arange(-h//2, h//2)
    grid_x, grid_y = np.meshgrid(xs, ys)  # 2d matrices

    # point = centre + (x * u_vec) + (y * v_vec)
    # reshape everything to (3,1)
    centre_col = np.array(center).reshape(3, 1)
    step_u_col = step_u.reshape(3, 1)
    step_v_col = step_v.reshape(3, 1)

    flat_x = grid_x.flatten()  # to 1D arrays to make the math easier
    flat_y = grid_y.flatten()

    coords = centre_col + (step_u_col * flat_x) + \
        (step_v_col * flat_y)  # (3,N) array
    return coords, (h, w)


def reslice(volume, coords, shape):  # interpolates the volume at given coordinates
    # nearest handles boundaries better
    samples = map_coordinates(volume, coords, order=1, mode='nearest')
    return samples.reshape(shape)


#FEATURE EXTRACTION
def extract_baseline_features(volume): #edge detection using sobel
    print("computing baseline features")
    gz = sobel(volume, axis=0)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=2)
    mag = np.sqrt(gz ** 2 + gy ** 2 + gz ** 2)
    return mag

def extract_new_features(volume): #difference of gaussians
    #define feature range
    sigma_inner = 2.0 #reduces speckle noise
    sigma_outer = 4.0 #range to keep

    gaussian_inner = gaussian_filter(volume, sigma_inner) #2x gaussian of the image
    gaussian_outer = gaussian_filter(volume, sigma_outer)
    dog = gaussian_inner - gaussian_outer

    feature_map = np.abs(dog) #absolute values to highlight boundaries
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) #normalise values

    threshold = np.percentile(feature_map, 85) #keep strongest 15%
    feature_map[feature_map < threshold] = 0
    return feature_map




#################################################################################################
if __name__ == "__main__":
    volume, spacing = load_data()

    print("testing reslicing")
    real_centre = np.array(volume.shape) // 2
    normals = {
        "axial (top view)":    (1.0, 0.0, 0.0),  # normal points in Z
        "coronal (front view)": (0.0, 1.0, 0.0),  # normal points in Y
        "sagittal (side view)": (0.0, 0.0, 1.0)  # normal points in X
    }

    plt.figure(figsize=(15, 5))

    for i, (name, n) in enumerate(normals.items()):
        print(f"generating {name} with normal {n}...")
        coords, shape_out = get_reslice_coordinates(
            real_centre, n, spacing, (256, 256))  # generate slice
        slice_img = reslice(volume, coords, shape_out)

        plt.subplot(1, 3, i+1)
        # origin='lower' often helps orientation
        plt.imshow(slice_img, cmap='gray', origin='lower')
        plt.title(f"{name}\nNormal: {n}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("testing feature extractors")
    #smaller_volume = np.random.rand(50, 50, 50).astype(np.float32)
    feature_baseline = extract_baseline_features(volume) #test baseline
    #print(f"baseline shape {feature_baseline.shape} (should be 50,50,50)")

    feature_new = extract_new_features(volume) #test hessian
    #print(f"new feature shape {feature_new.shape} (should be 50,50,50)")

    if feature_baseline.shape == volume.shape and feature_new.shape == volume.shape:
        print("shapes match")
    mid_slice = volume.shape[0] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(volume[mid_slice, :, :], cmap='gray')
    plt.title("original (real data)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(feature_baseline[mid_slice, :, :], cmap='hot')
    plt.title("baseline (sobel edge detection)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(feature_new[mid_slice, :, :], cmap='hot')
    plt.title("new (difference of gaussians)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()



    # real_normal = (1.0, 0.0, 1.0)
    # output_shape = (256, 256)
    # print(f"generating coordinates at {real_centre} and real normal {real_normal}")
    # coordinates, shape_out = get_reslice_coordinates(real_centre, real_normal, spacing, output_shape)
    # print(coordinates)
    # print(shape_out)

    # print("interpolating slice")
    # slice_img = reslice(volume, coordinates, shape_out)

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(slice_img, cmap='gray', aspect='equal')
    # plt.title(f"resliced view\nnormal: {real_normal}")
    # plt.axis('off')

    # plt.subplot(1,2,2)
    # plt.imshow(volume[int(real_centre[0]), :,:], cmap='gray')
    # plt.title("original axial slice z-plane")
    # plt.axis('off')
    # plt.show()
