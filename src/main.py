import torch
from rays import *
from sphere import *
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
    sphere1 = Sphere(torch.tensor([0,0,-1.3],device = device),0.5,torch.tensor([0.5, 0.5, 0.5], device=device), "diffuse")
    sphere2 = Sphere(torch.tensor([0,-100.5,-1],device = device),100,torch.tensor([0.5, 1.0, 0.5], device=device), "diffuse")
    sphere3 = Sphere(torch.tensor([1.2,-0.3,-1],device = device),0.25,torch.tensor([0.5, 0.5, 1.0], device=device), "metal")
    sphere4 = Sphere(torch.tensor([-1.2,-0.1,-1],device = device),0.5,torch.tensor([1.0, 0.5, 1.0], device=device), "metal")
    objectList = [sphere1,sphere2,sphere3,sphere4]
    for i in range(height):
        for j in range(width):
            currentPixel = topLeftPixel + (i*deltaV) + (j*deltaU)
            pixelTensor[i, j] = currentPixel    
    topLeftPixel = topLeftPixel.to(device)
    cameraOrigin = cameraOrigin.to(device)    
    pixelTensor = pixelTensor.to(device)
    deltaU = deltaU.to(device)
    deltaV = deltaV.to(device)
    originTensor = torch.empty_like(pixelTensor) 
    originTensor[:] = cameraOrigin
    rayTensor = pixelTensor - cameraOrigin
    colourTensor = torch.zeros((height, width, 3), device=device)
    samples = 500
    for i in range(samples):
        jitter = torch.rand(height, width, 2, device=device)
        jitteredPixel = topLeftPixel[None, None, :] + (torch.arange(height, device=device)[:, None, None] + jitter[..., 1:2]) * deltaV + (torch.arange(width, device=device)[None, :, None] + jitter[..., 0:1]) * deltaU
        rayTensor = jitteredPixel - originTensor
        colourTensor += colour(rayTensor, device, originTensor, objectList)
    colourTensor = colourTensor / samples
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
