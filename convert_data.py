import SimpleITK as sitk
import numpy as np

def convert():
    input_filename = "case001_trus.gipl"
    print(f"reading {input_filename}")

    try:
        image = sitk.ReadImage(input_filename)
    except RuntimeError:
        print(f"couldnt find {input_filename}")
        return
    
    #get data and spacing
    data = sitk.GetArrayFromImage(image)
    spacing = np.array(image.GetSpacing()[::-1]) #flip to z,y,x for numpy

    print(f"original size {data.shape}")
    print(f"original spacing {spacing}")

    np.save("image.npy", data)
    np.save("spacing.npy", spacing)
    print("created both files")


########################################################################################
if __name__ == "__main__":
    convert()