
import torch

def colour(rays, device, origins, objects):
    directions = torch.nn.functional.normalize(rays, dim=-1)
    colours = torch.zeros((*rays.shape[:-1], 3), device=device)
    throughput = torch.ones((*rays.shape[:-1], 3), device=device)

    for i in range(5):
        closest = torch.full(rays.shape[:2], float("inf"), device=device)
        closestNormals = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        closestPoints = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        closestColours = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        randomTensor = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        closestMaterial = torch.full(rays.shape[:2], 0.0, device=device)
        for obj in objects:   
            roots = obj.intersect(rays, origins)
            roots = roots.clamp(min=0.0004)
            points = origins + directions * roots.unsqueeze(-1)
            normals = torch.nn.functional.normalize(points - obj.center, dim=-1)
            mask = (roots > 1e-4) & (roots < closest) 
            closest = torch.where(mask, roots, closest)
            closestColours = torch.where(mask.unsqueeze(-1), obj.colour, closestColours)
            closestNormals = torch.where(mask.unsqueeze(-1), normals, closestNormals)
            closestPoints = torch.where(mask.unsqueeze(-1), points, closestPoints)
            match obj.material:
                case "diffuse":
                    closestMaterial = torch.where(mask, torch.full_like(closestMaterial, 0), closestMaterial)
                case "metal":
                    closestMaterial = torch.where(mask, torch.full_like(closestMaterial, 1), closestMaterial)
                 
        hitMask = closest < float("inf")
        a = 0.5 * (directions[:, :, 1] + 1)  
        background = (1.0 - a)[:, :, None] * torch.tensor([1.0, 1.0, 1.0], device=device) + a[:, :, None] * torch.tensor([0.5, 0.7, 1.0], device=device)
        colours = colours + throughput * torch.where(hitMask[..., None], torch.zeros_like(background), background)
        throughput = torch.where(hitMask[..., None], throughput, torch.zeros_like(throughput))
        albedo = closestColours
        randomTensor = torch.zeros_like(closestNormals)
        diffuseMask = (closestMaterial == 0) & hitMask
        if diffuseMask.any():
            diffuse = torch.nn.functional.normalize(torch.randn_like(closestNormals), dim=-1)
            diffuse = diffuse + closestNormals
            randomTensor = torch.where(diffuseMask.unsqueeze(-1), diffuse, randomTensor)
        metalMask = (closestMaterial == 1) & hitMask
        if metalMask.any():
            metal = directions - (2* torch.sum((directions)*(closestNormals), dim=-1, keepdim=True) * closestNormals)
            randomTensor = torch.where(metalMask.unsqueeze(-1), metal, randomTensor)
        origins = torch.where(hitMask[..., None], closestPoints + 1e-4 * closestNormals, origins)
        directions = randomTensor
        rays = directions  
        throughput = torch.where(hitMask[..., None], throughput * albedo, throughput)
    
    return colours