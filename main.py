import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, sobel, uniform_filter
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

###################################################################################################################################
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

###################################################################################################################################
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

    # choose arbitrary vector that is furthest from the parallel
    abs_normal = np.abs(normal)
    if abs_normal[0] <= abs_normal[1] and abs_normal[0] <= abs_normal[2]:
        arbitrary = np.array([1, 0, 0])
    elif abs_normal[1] <= abs_normal[2]:
        arbitrary = np.array([0, 1, 0])
    else:
        arbitrary = np.array([0, 0, 1])

    v_vec = np.cross(arbitrary, normal)  # cross product will lie flat on plane
    v_vec /= np.linalg.norm(v_vec)

    u_vec = np.cross(v_vec, normal)  # u_vec would be perpendicular to both
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
    samples = map_coordinates(volume, coords, order=1, mode='constant')
    return samples.reshape(shape)

###################################################################################################################################
# FEATURE EXTRACTION
def extract_baseline_features(volume, sigma=2.0): #laplacian of gaussian edge detection
    Izz = gaussian_filter(volume, sigma, order=[2, 0, 0]) #laplacian of gaussian = sum of second derivatives of Gaussian smoothed image
    Iyy = gaussian_filter(volume, sigma, order=[0, 2, 0])
    Ixx = gaussian_filter(volume, sigma, order=[0, 0, 2])
    
    log_response = Izz + Iyy + Ixx
    return np.abs(log_response) #take absolute values to get magnitude

# new feature extraction method
# structure tensor coherence
def extract_structure_tensor_features(volume, sigma_gradient=1.0, sigma_tensor=2.0):
    iz = gaussian_filter(volume, sigma_gradient, order=[1,0,0]) #smoothed gradients
    iy = gaussian_filter(volume, sigma_gradient, order=[0,1,0])
    ix = gaussian_filter(volume, sigma_gradient, order=[0,0,1])

    jxx = gaussian_filter(ix * ix, sigma_tensor) #smoothed outer products
    jyy = gaussian_filter(iy * iy, sigma_tensor)
    jzz = gaussian_filter(iz * iz, sigma_tensor)
    jxy = gaussian_filter(ix * iy, sigma_tensor)
    jxz = gaussian_filter(ix * iz, sigma_tensor)
    jyz = gaussian_filter(iy * iz, sigma_tensor)

    shape = volume.shape
    num_vox = np.prod(shape)
    J = np.zeros((num_vox, 3, 3)) #fill values 
    J[:, 0, 0] = jxx.flatten()
    J[:, 1, 1] = jyy.flatten()
    J[:, 2, 2] = jzz.flatten()
    J[:, 0, 1] = J[:, 1, 0] = jxy.flatten()
    J[:, 0, 2] = J[:, 2, 0] = jxz.flatten()
    J[:, 1, 2] = J[:, 2, 1] = jyz.flatten()

    eigs = np.linalg.eigvalsh(J) #sort
    lambda_1 = eigs[:,0] #smallest eigenvalue
    lambda_2 = eigs[:,1] #middle
    lambda_3 = eigs[:,2] #largest

    #surfaceness measure - lambda_3 > lambda_2
    surfaceness = (np.abs(lambda_3) - np.abs(lambda_2)) / (np.abs(lambda_3) + 1e-8)
    gradient_magnitude = np.abs(lambda_1) + np.abs(lambda_2) + np.abs(lambda_3) #overall strength to be added as weighting
    surfaceness = np.maximum(surfaceness, 0) #prevent non-negative values
    feature = surfaceness * np.sqrt(gradient_magnitude)
    feature = feature.reshape(shape)
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)    #handle nan cases   
    p1, p99 = np.percentile(feature, [1,99])
    feature = np.clip(feature, p1, p99)
    feature = (feature - p1) / (p99 - p1 + 1e-8)
    return feature




###################################################################################################################################
# EVALUATION METRIC - coherence edge ratio
def calculate_cer(volume, feature_map, threshold_percentile=85, gradient_sigma=1.0, window_size=5):
    # measures the consistency of gradients at detected feature locations - helps to ignore random gradients caused by speckle noise
    gz = gaussian_filter(volume, gradient_sigma, order=[1, 0, 0]) #smoothed gradientts of the original image
    gy = gaussian_filter(volume, gradient_sigma, order=[0, 1, 0])
    gx = gaussian_filter(volume, gradient_sigma, order=[0, 0, 1])
    
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-8 #get magnitude of gradient + epsilon to avoid divison by 0
    
    gx_unit = gx / grad_mag #unit gradient vectors
    gy_unit = gy / grad_mag
    gz_unit = gz / grad_mag
    
    # local coherence = average unit vectors in neighbourhood
    # expect that if the gradients point in the same direction - avg mag approx 1
    avg_gx = uniform_filter(gx_unit, window_size)
    avg_gy = uniform_filter(gy_unit, window_size)
    avg_gz = uniform_filter(gz_unit, window_size)
    
    # coherence = magnitude of averaged unit vectors
    local_coherence = np.sqrt(avg_gx**2 + avg_gy**2 + avg_gz**2)
    
    if feature_map.max() > 0: #threshold feature map
        feat_norm = feature_map / feature_map.max()
    else:
        return 0.0
    threshold = np.percentile(feat_norm, threshold_percentile)
    feature_mask = feat_norm > threshold
    
    if feature_mask.sum() == 0:
        return 0.0
    
    # CER = average coherence at detected feature locations
    cer = np.mean(local_coherence[feature_mask]) 
    return cer

def calculate_cer_2d(image, feature_map, threshold_percentile=85, gradient_sigma=1.0, window_size=5): #2D version - per slice analysis
    gy = gaussian_filter(image, gradient_sigma, order=[1, 0]) #smoothed gradients
    gx = gaussian_filter(image, gradient_sigma, order=[0, 1])
    
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-8 #gradient magnitude to normalise vectors and prevent division by 0
    
    gx_unit = gx / grad_mag #normalise to exttract direction not strength
    gy_unit = gy / grad_mag
    
    avg_gx = uniform_filter(gx_unit, window_size) #vector averaging to compute local coherence
    avg_gy = uniform_filter(gy_unit, window_size)
    local_coherence = np.sqrt(avg_gx**2 + avg_gy**2) #magnitude
    
    if feature_map.max() > 0: #filtering
        feat_norm = feature_map / feature_map.max()
    else:
        return 0.0
    
    threshold = np.percentile(feat_norm, threshold_percentile) #top 15% brightest pixels
    feature_mask = feat_norm > threshold
    
    if feature_mask.sum() == 0:
        return 0.0
    return np.mean(local_coherence[feature_mask])

def statistical_comparison(volume, baseline_features, new_features, num_slices=None): #paired ttest to see if there is a significant difference
    if num_slices is None: #use all slices
        num_slices = volume.shape[0]
    slice_indices = np.linspace(2, volume.shape[0] - 3, num_slices, dtype=int) #sample slice indices, skipping edge slice cases
    
    baseline_cer_scores = []
    new_cer_scores = []
    
    for z in slice_indices: #loop through slices
        # extract 2D slices
        vol_slice = volume[z, :, :]
        base_slice = baseline_features[z, :, :]
        new_slice = new_features[z, :, :]
        
        # compute 2D CER for each slice
        cer_base = calculate_cer_2d(vol_slice, base_slice)
        cer_new = calculate_cer_2d(vol_slice, new_slice)
        
        baseline_cer_scores.append(cer_base)
        new_cer_scores.append(cer_new)
    
    baseline_cer_scores = np.array(baseline_cer_scores) #convert to numpy
    new_cer_scores = np.array(new_cer_scores)
    
    # paired t-test (same slices, different methods)
    t_stat, p_value = stats.ttest_rel(new_cer_scores, baseline_cer_scores)
    
    results = {
        'baseline_mean': np.mean(baseline_cer_scores),
        'baseline_std': np.std(baseline_cer_scores),
        'new_mean': np.mean(new_cer_scores),
        'new_std': np.std(new_cer_scores),
        't_statistic': t_stat,
        'p_value': p_value,
        'num_slices': len(slice_indices),
        'baseline_scores': baseline_cer_scores,
        'new_scores': new_cer_scores
    }
    return results

###################################################################################################################################
def experiment_vary_parameters(volume, spacing):
    print("varying resclicing parameters")
    centre = np.array(volume.shape)//2
    # define 3 different angles - normal, ~30deg, ~60deg
    normals = [
        (0.0, 0.0, 1.0),  # standard
        (0.0, 0.4, 0.9),  # reasonable
        (0.0, 0.8, 0.6)  # steep
    ]
    plt.figure(figsize=(15, 6))
    for i, norm in enumerate(normals):
        coords, shape = get_reslice_coordinates(
            centre, norm, spacing, (256, 256))
        slice_img = reslice(volume, coords, shape)
        plt.subplot(1, 3, i+1)
        plt.title(f"normal: {norm}\n(tilt {i*30}) deg")
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')
    plt.savefig("exp_1_varying_angles.png")
    print("saved varying angles png")


def experiment_feature_comparison(volume, baseline, new_features, spacing):
    print("experiment for visual comparison and overlays")
    z_idx = volume.shape[0] // 2  # middle of volume
    ortho_img = volume[z_idx, :, :]
    ortho_base = baseline[z_idx, :, :]
    ortho_new = new_features[z_idx, :, :]

    centre = np.array(volume.shape) // 2
    norm = (0.0, 0.4, 0.9)  # "reasonable" angle
    coords, shape = get_reslice_coordinates(centre, norm, spacing, (256, 256))

    reslice_img = reslice(volume, coords, shape)
    reslice_base = reslice(baseline, coords, shape)
    reslice_new = reslice(new_features, coords, shape)

    # plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(ortho_img, cmap='gray')
    plt.title("orthogonal: ultrasound")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(ortho_img, cmap='gray')
    plt.imshow(ortho_base, cmap='hot', alpha=0.5)  # overlay
    plt.title("overlay: baseline (Laplacian of Gaussian)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(ortho_img, cmap='gray')
    plt.imshow(ortho_new, cmap='hot', alpha=0.5)
    plt.title("overlay: new structure tensor")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(reslice_img, cmap='gray')
    plt.title("resliced: ultrasound")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(reslice_img, cmap='gray')
    plt.imshow(reslice_base, cmap='hot', alpha=0.5)
    plt.title("overlay: baseline")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(reslice_img, cmap='gray')
    plt.imshow(reslice_new, cmap='hot', alpha=0.5)
    plt.title("overlay: new structure tensor")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("exp_2_overlays2d.png")
    print("saved exp2")


def experiment_3d_visualisation(volume, baseline, new_features, spacing): #stack of parallel axial slices with feature overlays
    nz, ny, nx = volume.shape
    slice_indices = [int(nz * 0.3), int(nz * 0.5), int(nz * 0.7)] #3 slices chosen
    z_aspect = spacing[0] / spacing[1] #scale z to match proportions of x/y
    
    fig = plt.figure(figsize=(14, 6))
    
    def normalise(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    def create_overlay(vol_slice, feature_slice, alpha=0.6):
        #alpha controls transparency
        vol_norm = normalise(vol_slice)
        feat_norm = normalise(feature_slice)
        
        rgba = np.zeros((*vol_slice.shape, 4)) #create rgb channels
        
        rgba[..., 0] = np.clip(vol_norm + feat_norm * alpha, 0, 1) #r = volume + feature
        rgba[..., 1] = np.clip(vol_norm * (1 - feat_norm * 0.5), 0, 1) #g = volume
        rgba[..., 2] = np.clip(vol_norm * (1 - feat_norm * 0.8), 0, 1) #b = volume
        
        rgba[..., 3] = 0.9 
        return rgba

    # baseline plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D axial stack: baseline (Laplacian of Gaussian)")
    
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny)) #meshgrid for the slice plane (x,y)
    
    for z_idx in slice_indices:
        z_plane = np.ones_like(x_grid) * z_idx #create z plane at specific index
        
        vol_slice = volume[z_idx, :, :]
        feat_slice = baseline[z_idx, :, :]
        colours = create_overlay(vol_slice, feat_slice)

        ax1.plot_surface(x_grid, y_grid, z_plane, facecolors=colours, shade=False, rstride=2, cstride=2)

    # aspect ratio set
    ax1.set_box_aspect((1, 1, z_aspect * nz/nx)) # scale z axis to fit x/y
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('slice index')
    ax1.view_init(elev=30, azim=-60) # Good angle to see the stack

    # structure tensor plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("3D axial stack: new structure tensor extraction")
    
    for z_idx in slice_indices:
        z_plane = np.ones_like(x_grid) * z_idx
        
        vol_slice = volume[z_idx, :, :]
        feat_slice = new_features[z_idx, :, :]
        
        colors = create_overlay(vol_slice, feat_slice)
        ax2.plot_surface(x_grid, y_grid, z_plane, facecolors=colors, shade=False, rstride=2, cstride=2)

    ax2.set_box_aspect((1, 1, z_aspect * nz/nx))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('slice index')
    ax2.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig("exp_3_3d_stack_visualisation.png", dpi=150)
    print("saved 3D stack visualisation")

###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    volume, spacing = load_data()  # load data
    # normalise values for consistent processing
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    # extract features
    baseline_features = extract_baseline_features(volume)
    new_features = extract_structure_tensor_features(volume)

    # experiments
    experiment_vary_parameters(volume, spacing)
    experiment_feature_comparison(
        volume, baseline_features, new_features, spacing)
    experiment_3d_visualisation(volume, baseline_features, new_features, spacing)

    # define reslicing parameters
    centre = np.array(volume.shape) // 2  # centre of the volume
    norm = (0.0, 0.4, 0.9)  # NON-orthogonal normal -- tilted to 40deg
    print(f"re-slicing at {centre}, normal {norm}")

    coordinates, shape = get_reslice_coordinates(
        centre, norm, spacing, (256, 256))  # generate coordinates

    # re-slice images
    image_resliced = reslice(volume, coordinates, shape)
    baseline_resliced = reslice(baseline_features, coordinates, shape)
    new_resliced = reslice(new_features, coordinates, shape)

    baseline_cer = calculate_cer(volume, baseline_features)
    new_cer = calculate_cer(volume, new_features)

    print(f"baseline CER score: {baseline_cer:.4f}")
    print(f"new method CER score: {new_cer:.4f}")
    if new_cer > baseline_cer:
        print(f"success improved contrast by {new_cer/baseline_cer:.1f}x")
    else:
        print("check parameters")

    print("\n--- stats analysis ---")
    stat_results = statistical_comparison(volume, baseline_features, new_features)
    
    print(f"baseline CER: {stat_results['baseline_mean']:.4f} ± {stat_results['baseline_std']:.4f}")
    print(f"new structure tensor extraction method CER: {stat_results['new_mean']:.4f} ± {stat_results['new_std']:.4f}")
    print(f"paired t-test: t={stat_results['t_statistic']:.3f}, p={stat_results['p_value']:.4e}")
    
    if stat_results['p_value'] < 0.05:
        print("result: statistically significant difference (p < 0.05)")
    else:
        print("result: no statistically significant difference")

    # plotting
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_resliced, cmap='gray')
    plt.title("re-sliced ultrasound")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(baseline_resliced, cmap='hot')
    plt.title("baseline (smoothed Laplacian of Gaussian)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(new_resliced, cmap='hot')
    plt.title("new (structure tensor)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("report_figure.png")
    print("processing complete. saved 'report_figure.png'.")

