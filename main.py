import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, sobel

# data loading and pre-processing


def load_data(file_path):  # loads medical data
    if not os.path.exists(file_path):
        print(f"error -  file at {file_path} not found")
        sys.exit(1)
    print(f"loading file from {file_path}")

    if file_path.endswith('.npy'):
        data = np.load(file_path)
        spacing = (0.5, 0.5, 0.5)
    else:
        image = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(image)

        # flip spacing to numpy format
        spacing = image.GetSpacing()[::-1]

    print(f"data loaded, shape {data.shape}, spacing {spacing}")
    return data.astype(np.float32), spacing


#################################################################################################
if __name__ == "__main__":
    filename = "case001_trus.gipl"

    try:
        volume, spacing = load_data(filename)  # run the load function
        print("success - plotting middle slice")

        middle_z = volume.shape[0] // 2

        plt.figure(figsize=(6, 6))
        plt.imshow(volume[middle_z, :, :], cmap='gray')
        plt.title("middle slice")
        plt.axis('off')
        plt.show()

    except SystemExit:
        pass
    except Exception as pe:
        print("unexpected error occurred")
