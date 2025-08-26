import torch
from rays import *
import os
def write_color(output, colour):
    r = colour[0];
    g = colour[1];
    b = colour[2];

    r = int(255 * r);
    g = int(255 * g);
    b = int(255 * b);
    output.write(f"{r} {g} {b} ")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    ratio = 16 / 9
    width = 400
    height = int(width/ratio)
    if height < 1: height = 1
    focalLength = 1
    viewportHeight = 2
    viewportWidth = viewportHeight * (width/height)
    u = torch.tensor([viewportWidth,0,0])
    v = torch.tensor([0,-viewportHeight,0])
    deltaU = u/width
    deltaV = v/height
    cameraOrigin = torch.tensor([0,0,0])
    topLeft = cameraOrigin - torch.tensor([0,0,focalLength]) - u/2 -v/2
    topLeftPixel = topLeft + 0.5*(deltaU+deltaV)
    pixelTensor = torch.empty(height,width,3)
    for i in range(height):
        for j in range(width):
            currentPixel = topLeftPixel + (i*deltaV) + (j*deltaU)
            pixelTensor[i, j] = currentPixel    
    cameraOrigin = cameraOrigin.to(device)     
    pixelTensor = pixelTensor.to(device)
    rayTensor = pixelTensor - cameraOrigin
    colourTensor = colour(rayTensor,device)
    img = (colourTensor.clamp(0,1) * 255).to(torch.uint8).cpu()
    with open("output/image.ppm", "w") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for row in img:
            for pixel in row:
                r, g, b = pixel.tolist()
                f.write(f"{r} {g} {b} ")
            f.write("\n")
if __name__ == "__main__":
    main()
