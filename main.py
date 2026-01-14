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

def extract_new_features(volume): #difference of eigenvalues via hessian
    # blobs are supressed with sheets enhanced (lambda_3 > lambda_2)
    sigma = 2.5 #initial gaussian smoothing
    # img_smooth =  gaussian_filter(volume, sigma)

    #hessian components
    Izz = gaussian_filter(volume, sigma, order=[2, 0, 0], mode='nearest') #mode=nearest to avoid boundary artifacts
    Iyy = gaussian_filter(volume, sigma, order=[0, 2, 0], mode='nearest')
    Ixx = gaussian_filter(volume, sigma, order=[0, 0, 2], mode='nearest')
    Izy = gaussian_filter(volume, sigma, order=[1, 1, 0], mode='nearest')
    Izx = gaussian_filter(volume, sigma, order=[1, 0, 1], mode='nearest')
    Iyx = gaussian_filter(volume, sigma, order=[0, 1, 1], mode='nearest')

    shape = volume.shape #(n_pixels, 3, 3)
    num_pixels = np.prod(shape)
    H = np.zeros((num_pixels, 3, 3)) #hessian matrix
    #assign values 
    H[:, 0, 0] = Izz.flatten() 
    H[:, 1, 1] = Iyy.flatten()
    H[:, 2, 2] = Ixx.flatten()
    H[:, 0, 1] = H[:, 1, 0] = Izy.flatten()
    H[:, 0, 2] = H[:, 2, 0] = Izx.flatten()
    H[:, 1, 2] = H[:, 2, 1] = Iyx.flatten()

    evals = np.linalg.eigvalsh(H) #compute eigenvalues

    idx = np.argsort(np.abs(evals), axis=1) #sorting array to sort eigenvalues by magnitude
    evals_sorted = np.take_along_axis(evals, idx, axis=1) #reorder eigenvalues using indices

    # extract eigenvalues
    e1 = evals_sorted[:,0] #smallest eigenvalue (curvature)
    e2 = evals_sorted[:,1] #medium curvature
    e3 = evals_sorted[:,2] #largest curvature (normal)

    # sheetness = difference between largest and medium curvature - reduces blob from structure
    flat_sheetness = np.abs(e3) - np.abs(e2)
    flat_sheetness[flat_sheetness < 0] = 0
    map_sheetness = flat_sheetness.reshape(shape) #back to 3D volume

    v_min, v_max = np.percentile(map_sheetness, [1, 99])
    map_sheetness = np.clip(map_sheetness, v_min, v_max)
    map_sheetness = (map_sheetness - v_min) / (v_max - v_min + 1e-8) #normalisation
    map_sheetness = map_sheetness ** 2.0

    # threshold = np.percentile(map_sheetness, 85) #thresholding by keeping top 15%
    # map_sheetness[map_sheetness < threshold] = 0
    return map_sheetness

    # define feature range
    # sigma_inner = 2.0 #reduces speckle noise
    # sigma_outer = 4.0 #range to keep

    # gaussian_inner = gaussian_filter(volume, sigma_inner) #2x gaussian of the image
    # gaussian_outer = gaussian_filter(volume, sigma_outer)
    # dog = gaussian_inner - gaussian_outer

    # feature_map = np.abs(dog) #absolute values to highlight boundaries
    # feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) #normalise values

    # threshold = np.percentile(feature_map, 85) #keep strongest 15%
    # feature_map[feature_map < threshold] = 0

    # mask = feature_map > threshold 

    # clean_mask = binary_opening(mask, structure=np.ones((2,2,2))).astype(np.float32) #binary structure defined for morphological opening
    # final_output = gaussian_filter(clean_mask, sigma=0.5) #extra smoothing via gaussian
    # return clean_mask
    # return feature_map


# EVALUATION - contrast to noise ratio
def calculate_cnr(image, feature_map): # CNR = |mean_in - mean_out| / std_out
    if feature_map.max() > 0:
        feature_map = feature_map / feature_map.max() #normalisation
    #create binary mask using top 15% of values
    threshold = np.percentile(feature_map, 85)
    mask = feature_map > threshold

    if mask.sum() == 0 or mask.sum() == mask.size:
        return 0.0
        
    inside_mean = np.mean(image[mask])
    outside_mean = np.mean(image[~mask])
    outside_std = np.std(image[~mask])
    
    if outside_std < 1e-6: #avoid division by zero
        return 0.0
    
    return np.abs(inside_mean - outside_mean) / outside_std


#experiments##############################################################################################
def experiment_vary_parameters(volume, spacing):
    print("varying resclicing parameters")
    centre = np.array(volume.shape)//2

    #define 3 different angles - normal, ~30deg, ~60deg
    normals = [
        (0.0, 0.0, 1.0),    #standard
        (0.0, 0.4, 0.9),    #reasonable
        (0.0, 0.8, 0.6)     #steep
    ]

    plt.figure(figsize=(15,6))
    for i, norm in enumerate(normals):
        coords, shape = get_reslice_coordinates(centre, norm, spacing, (256, 256))
        slice_img = reslice(volume, coords, shape)
        plt.title(f"normal: {norm}\n(tilt {i*30}) deg")
        plt.axis('off')
    plt.savefig("exp_1_varying_angles.png")
    print("saved varying angles png")

def experiment_feature_comparison(volume, baseline, new_features, spacing):
    print("experiment for visual comparison and overlays")
    z_idx = volume.shape[0] // 2 #middle of volume
    ortho_img  = volume[z_idx, :, :]
    ortho_base = baseline[z_idx, :, :]
    ortho_new = new_features[z_idx, :, :]

    centre = np.array(volume.shape) // 2
    norm = (0.0, 0.4, 0.9) #"reasonable" angle
    coords, shape = get_reslice_coordinates(centre, norm, spacing, (256,256))

    reslice_img = reslice(volume, coords, shape)
    reslice_base = reslice(baseline, coords, shape)
    reslice_new = reslice(new_features, coords, shape)

    #plotting
    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1)
    plt.imshow(ortho_img, cmap='gray')
    plt.title("orthogonal: ultrasound")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(ortho_img, cmap='gray')
    plt.imshow(ortho_base, cmap='hot', alpha=0.5) #overlay
    plt.title("overlay: baseline (sobel)")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(ortho_img, cmap='gray')
    plt.imshow(ortho_new, cmap='hot', alpha=0.5)
    plt.title("overlay: new (sheetness)")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(reslice_img, cmap='gray')
    plt.title("resliced: ultrasound")
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(reslice_img, cmap='gray')
    plt.imshow(reslice_base, cmap='hot', alpha=0.5)
    plt.title("overlay: baseline")
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.imshow(reslice_img, cmap='gray')
    plt.imshow(reslice_new, cmap='hot', alpha=0.5)
    plt.title("overlay: new")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("exp_2_overlays.png")
    print("saved exp2")



















#main##########################################################################################################################################################
if __name__ == "__main__":
    volume, spacing = load_data() #load data
    volume = (volume - volume.min()) / (volume.max() - volume.min()) #normalise values for consistent processing

    #extract features
    baseline_features = extract_baseline_features(volume)
    new_features = extract_new_features(volume)

    #define reslicing parameters
    centre = np.array(volume.shape) // 2 #centre of the volume
    norm = (1.0, 0.4, 0.0) #NON-orthogonal normal -- tilted to 40deg
    print(f"re-slicing at {centre}, normal {norm}")

    coordinates, shape = get_reslice_coordinates(centre, norm, spacing, (256, 256)) #generate coordinates

    #re-slice images
    image_resliced = reslice(volume, coordinates, shape)
    baseline_resliced = reslice(baseline_features, coordinates, shape)
    new_resliced = reslice(new_features, coordinates, shape)

    baseline_cnr = calculate_cnr(baseline_features, baseline_features)
    new_cnr = calculate_cnr(new_features, new_features)
    
    print(f"baseline CNR score: {baseline_cnr:.4f}")
    print(f"new method CNR score: {new_cnr:.4f}")
    if new_cnr > baseline_cnr:
        print(f"success improved contrast by {new_cnr/baseline_cnr:.1f}x")
    else:
        print("check parameters")

    # plotting   
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_resliced, cmap='gray')
    plt.title("re-sliced ultrasound")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(baseline_resliced, cmap='hot')
    plt.title("baseline (sobel edge detection)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(new_resliced, cmap='hot')
    plt.title("new (difference of eigenvalues)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("report_figure.png")
    print("processing complete. saved 'report_figure.png'.")












    # volume, spacing = load_data()

    # print("testing reslicing")
    # real_centre = np.array(volume.shape) // 2
    # normals = {
    #     "axial (top view)":    (1.0, 0.0, 0.0),  # normal points in Z
    #     "coronal (front view)": (0.0, 1.0, 0.0),  # normal points in Y
    #     "sagittal (side view)": (0.0, 0.0, 1.0)  # normal points in X
    # }

    # plt.figure(figsize=(15, 5))

    # for i, (name, n) in enumerate(normals.items()):
    #     print(f"generating {name} with normal {n}...")
    #     coords, shape_out = get_reslice_coordinates(
    #         real_centre, n, spacing, (256, 256))  # generate slice
    #     slice_img = reslice(volume, coords, shape_out)

    #     plt.subplot(1, 3, i+1)
    #     # origin='lower' often helps orientation
    #     plt.imshow(slice_img, cmap='gray', origin='lower')
    #     plt.title(f"{name}\nNormal: {n}")
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    # print("testing feature extractors")
    # #smaller_volume = np.random.rand(50, 50, 50).astype(np.float32)
    # feature_baseline = extract_baseline_features(volume) #test baseline
    # #print(f"baseline shape {feature_baseline.shape} (should be 50,50,50)")

    # feature_new = extract_new_features(volume) #test hessian
    # #print(f"new feature shape {feature_new.shape} (should be 50,50,50)")

    # if feature_baseline.shape == volume.shape and feature_new.shape == volume.shape:
    #     print("shapes match")
    # mid_slice = volume.shape[0] // 2
    
    # plt.figure(figsize=(15, 5))
    
    # plt.subplot(1, 3, 1)
    # plt.imshow(volume[mid_slice, :, :], cmap='gray')
    # plt.title("original (real data)")
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(feature_baseline[mid_slice, :, :], cmap='hot')
    # plt.title("baseline (sobel edge detection)")
    # plt.axis('off')
    
    # plt.subplot(1, 3, 3)
    # plt.imshow(feature_new[mid_slice, :, :], cmap='hot')
    # plt.title("new (difference of gaussians)")
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()



    # # real_normal = (1.0, 0.0, 1.0)
    # # output_shape = (256, 256)
    # # print(f"generating coordinates at {real_centre} and real normal {real_normal}")
    # # coordinates, shape_out = get_reslice_coordinates(real_centre, real_normal, spacing, output_shape)
    # # print(coordinates)
    # # print(shape_out)

    # # print("interpolating slice")
    # # slice_img = reslice(volume, coordinates, shape_out)

    # # plt.figure(figsize=(10,5))
    # # plt.subplot(1,2,1)
    # # plt.imshow(slice_img, cmap='gray', aspect='equal')
    # # plt.title(f"resliced view\nnormal: {real_normal}")
    # # plt.axis('off')

    # # plt.subplot(1,2,2)
    # # plt.imshow(volume[int(real_centre[0]), :,:], cmap='gray')
    # # plt.title("original axial slice z-plane")
    # # plt.axis('off')
    # # plt.show()
