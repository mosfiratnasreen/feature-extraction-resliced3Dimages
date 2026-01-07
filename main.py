import os 
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, sobel

##data loading and pre-processing
def load_data(file_path): #loads medical data
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

        #flip spacing to numpy format
        spacing = image.GetSpacing()[::-1]

    print(f"data loaded, shape {data.shape}, spacing {spacing}")
    return data.astype(np.float32), spacing




