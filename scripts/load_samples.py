import os
import numpy as np
from PIL import Image

def main():
    npz_path = "/ccs/home/adityatomar/improved-diffusion/samples/samples_32x32x32x3.npz"
    output_path = "/ccs/home/adityatomar/improved-diffusion/samples/sample_images/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with np.load(npz_path) as f:
        arr = f["arr_0"]

    print(f"num samples: {arr.shape[0]}")
    print(f"dims: {arr.shape[1]}x{arr.shape[2]}x{arr.shape[3]}")

    for i in range(arr.shape[0]):
        im = Image.fromarray(arr[i])
        im.save(output_path + f"sample_{i}.png")

if __name__ == "__main__":
    main()
