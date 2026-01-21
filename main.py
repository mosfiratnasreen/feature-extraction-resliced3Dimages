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
        # fallback to simpleITK if needed
        try:
            import SimpleITK as sitk
            print(f"npy files not found - loading .gipl")
            img = sitk.ReadImage('case001_trus.gipl')
            volume = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = np.array(img.GetSpacing()[::-1])
        except (ImportError, RuntimeError):
            print("error could not load data")
            sys.exit(1)
    # print(f"volume shape {volume.shape}")
    # print(f"voxel spacing {spacing}")
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

    eigs = np.linalg.eigvalsh(J) # eigen value decomposition + sort
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
# VISUALISATION
def exp_varying_parameters(volume, spacing): #reslicing at non-orthogonal angles both in 2D and 3D
    nz, ny, nx = volume.shape
    z_aspect = spacing[0] / spacing[1] 
    centre = np.array(volume.shape) // 2
    
    normals = [
        (0.1, 0.1, 0.98),  # slight tilt (~10°) - ensure it is non orthogonal 
        (0.1, 0.4, 0.9),   # moderate tilt (~24°)
        (0.1, 0.7, 0.6)    # steep tilt (~50°)
    ]
    labels = ["Slight tilt", "Moderate tilt", "Steep tilt"]

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1.5])

    # 2D on top
    subfigs[0].suptitle('A. 2D Reslicing Results', fontsize=14, weight='bold')
    axs_2d = subfigs[0].subplots(1, 3)
    
    for i, norm in enumerate(normals):
        coords, shape = get_reslice_coordinates(centre, norm, spacing, (256, 256))
        slice_img = reslice(volume, coords, shape)
        
        axs_2d[i].imshow(slice_img, cmap='gray', aspect='equal')
        axs_2d[i].set_title(f"{labels[i]}\nNormal: {norm}", fontsize=10)
        axs_2d[i].set_xticks([]) #keep scale
        axs_2d[i].set_yticks([])

    # 3d plots
    subfigs[1].suptitle('B. 3D Plane Orientation', fontsize=14, weight='bold')
    axs_3d = [subfigs[1].add_subplot(1, 3, i+1, projection='3d') for i in range(3)]

    for i, norm in enumerate(normals):
        axs_3d[i].plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [0, 0, 0, 0, 0], 'k-', lw=0.5, alpha=0.5)
        axs_3d[i].plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [nz, nz, nz, nz, nz], 'k-', lw=0.5, alpha=0.5)
        axs_3d[i].plot([0, 0], [0, 0], [0, nz], 'k-', lw=0.5, alpha=0.5)
        axs_3d[i].plot([nx, nx], [ny, ny], [0, nz], 'k-', lw=0.5, alpha=0.5)
        axs_3d[i].plot([0, 0], [ny, ny], [0, nz], 'k-', lw=0.5, alpha=0.5)
        axs_3d[i].plot([nx, nx], [0, 0], [0, nz], 'k-', lw=0.5, alpha=0.5)

        coords, (h, w) = get_reslice_coordinates(centre, norm, spacing, (150, 150)) #extract and plot reslice plane
        X = coords[2].reshape(h, w)
        Y = coords[1].reshape(h, w)
        Z = coords[0].reshape(h, w)
        
        slice_img = reslice(volume, coords, (h, w))

        slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) #normalise for display
        colors = plt.cm.gray(slice_norm)
        
        # plot
        axs_3d[i].plot_surface(X, Y, Z, facecolors=colors, shade=False, 
                               rstride=5, cstride=5, antialiased=True)
        
        axs_3d[i].set_xlim(0, nx)
        axs_3d[i].set_ylim(0, ny)
        axs_3d[i].set_zlim(0, nz)
        axs_3d[i].set_title(labels[i], fontsize=11)
        
        axs_3d[i].set_xlabel('X')
        axs_3d[i].set_ylabel('Y')
        axs_3d[i].set_zlabel('Z')
        
        axs_3d[i].set_box_aspect((1, 1, z_aspect * (nz/nx))) 
        axs_3d[i].view_init(elev=20, azim=-45)
        
        axs_3d[i].xaxis.pane.fill = False
        axs_3d[i].yaxis.pane.fill = False
        axs_3d[i].zaxis.pane.fill = False
        axs_3d[i].grid(True)

    plt.savefig("exp_varying_parameters.png", dpi=300, bbox_inches='tight')
    print("Saved exp_varying_parameters.png")

#feature overlay on orthogonal and resliced planes in both 2d and 3d
def exp_feature_overlay_varying_angles(volume, baseline, new_features, spacing):
    nz, ny, nx = volume.shape
    z_aspect = spacing[0] / spacing[1]
    centre = np.array(volume.shape) // 2
    
    norm = (0.0, 0.4, 0.9) #clinically relevant tilt
    coords_reslice, shape_reslice = get_reslice_coordinates(centre, norm, spacing, (200, 200))

    def create_overlay(vol, feat):
        v = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        f = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        rgba = np.zeros((*vol.shape, 4))
        rgba[..., 0] = np.clip(v + f*0.9, 0, 1) # Red
        rgba[..., 1] = np.clip(v * (1 - f*0.5), 0, 1) 
        rgba[..., 2] = np.clip(v * (1 - f*0.5), 0, 1) 
        rgba[..., 3] = 1.0 
        return rgba

    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1.2])

    subfigs[0].suptitle('A. 2D Feature Extraction Comparison', fontsize=14, weight='bold')
    axs_2d = subfigs[0].subplots(1, 4)
    
    # orthogonal slices
    z_mid = nz // 2
    vol_ortho = volume[z_mid, :, :]
    base_ortho = baseline[z_mid, :, :]
    new_ortho = new_features[z_mid, :, :]
    
    axs_2d[0].imshow(create_overlay(vol_ortho, base_ortho), origin='lower')
    axs_2d[0].set_title("Orthogonal\nBaseline", fontsize=10)
    axs_2d[1].imshow(create_overlay(vol_ortho, new_ortho), origin='lower')
    axs_2d[1].set_title("Orthogonal\Structure Tensor", fontsize=10)

    # resliced slices
    vol_res = reslice(volume, coords_reslice, shape_reslice)
    base_res = reslice(baseline, coords_reslice, shape_reslice)
    new_res = reslice(new_features, coords_reslice, shape_reslice)
    
    axs_2d[2].imshow(create_overlay(vol_res, base_res), origin='lower')
    axs_2d[2].set_title("Resliced Non-orthogonal\nBaseline", fontsize=10)
    axs_2d[3].imshow(create_overlay(vol_res, new_res), origin='lower')
    axs_2d[3].set_title("Resliced Non-orthogonal\Structure Tensor", fontsize=10)
    
    for ax in axs_2d: ax.axis('off')

    subfigs[1].suptitle('B. 3D Plane Visualisation', fontsize=14, weight='bold')
    axs_3d = [subfigs[1].add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
    
    titles = ["Orthogonal Baseline", "Orthogonal Structure Tensor", "Resliced Non-orthogonal Baseline", "Resliced Non-orthogonal Structure Tensor"]
    
    # coordinates for resliced planes
    Xr = coords_reslice[2].reshape(shape_reslice)
    Yr = coords_reslice[1].reshape(shape_reslice)
    Zr = coords_reslice[0].reshape(shape_reslice)

    # coordinates for orthogonal planes
    x_ax, y_ax = np.meshgrid(np.arange(nx), np.arange(ny))
    z_ax = np.ones_like(x_ax) * z_mid

    for i in range(4):
        curr_feat = baseline if (i % 2 == 0) else new_features
    
        axs_3d[i].plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [0, 0, 0, 0, 0], 'k-', lw=0.5, alpha=0.3) #ensure wireframe
        axs_3d[i].plot([0, nx, nx, 0, 0], [0, 0, ny, ny, 0], [nz, nz, nz, nz, nz], 'k-', lw=0.5, alpha=0.3)
        axs_3d[i].plot([0, 0], [0, 0], [0, nz], 'k-', lw=0.5, alpha=0.3)
        axs_3d[i].plot([nx, nx], [ny, ny], [0, nz], 'k-', lw=0.5, alpha=0.3)

        if i < 2: 
            # orthogonal plane
            ov = create_overlay(volume[z_mid], curr_feat[z_mid])
            axs_3d[i].plot_surface(x_ax, y_ax, z_ax, facecolors=ov, shade=False, rstride=5, cstride=5)
        else: 
            # resliced plane
            vol_r = reslice(volume, coords_reslice, shape_reslice)
            feat_r = reslice(curr_feat, coords_reslice, shape_reslice)
            ov = create_overlay(vol_r, feat_r)
            axs_3d[i].plot_surface(Xr, Yr, Zr, facecolors=ov, shade=False, rstride=5, cstride=5)

        axs_3d[i].set_title(titles[i], fontsize=10)
        axs_3d[i].set_xlim(0, nx); axs_3d[i].set_ylim(0, ny); axs_3d[i].set_zlim(0, nz)
        axs_3d[i].set_box_aspect((1, 1, z_aspect * (nz/nx)))
        axs_3d[i].view_init(elev=25, azim=-50)

        axs_3d[i].xaxis.pane.fill = False
        axs_3d[i].yaxis.pane.fill = False
        axs_3d[i].zaxis.pane.fill = False
        axs_3d[i].grid(True)

    plt.savefig("exp_feature_overlay_varying_angles.png", dpi=300, bbox_inches='tight')
    print("Saved exp_feature_overlay_varying_angles.png")

###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    volume, spacing = load_data() #load data
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    #extract features - both baseline and structure tensor
    print("extracting features")
    baseline_features = extract_baseline_features(volume)
    new_features = extract_structure_tensor_features(volume)

    #generate plots
    exp_varying_parameters(volume, spacing)
    exp_feature_overlay_varying_angles(volume, baseline_features, new_features, spacing)

    # 4. Stats
    print("\n--- quantitative analysis ---")
    b_cer = calculate_cer(volume, baseline_features)
    n_cer = calculate_cer(volume, new_features)
    print(f"Global Baseline CER: {b_cer:.4f}")
    print(f"Global New Method CER: {n_cer:.4f}")

    stat_results = statistical_comparison(volume, baseline_features, new_features)
    print(f"Paired t-test: t={stat_results['t_statistic']:.3f}, p={stat_results['p_value']:.4e}")
    if stat_results['p_value'] < 0.05:
        print("result is statistically significant difference (p < 0.05)")
    else:
        print("result is not statistically significant")
    
    print("processing complete")
