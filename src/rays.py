
import torch

def colour(rays, device, origins, objects):
    directions = torch.nn.functional.normalize(rays, dim=-1)
    colours = torch.zeros((*rays.shape[:-1], 3), device=device)
    throughput = torch.ones((*rays.shape[:-1], 3), device=device)

    for i in range(15):
        closest = torch.full(rays.shape[:2], float("inf"), device=device)
        closestNormals = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        closestPoints = torch.full((*rays.shape[:2], 3), 0.0, device=device)
        for obj in objects:
            roots = obj.intersect(rays, origins)
            points = origins + directions * roots.unsqueeze(-1)
            normals = torch.nn.functional.normalize(points - obj.center, dim=-1)
            mask = roots < closest
            closest = torch.where(mask, roots, closest)
            closestNormals = torch.where(mask.unsqueeze(-1), normals, closestNormals)
            closestPoints = torch.where(mask.unsqueeze(-1), points, closestPoints)
        hitMask = closest < float("inf")
        if (~hitMask).any():
            a = 0.5 * (directions[:, :, 1] + 1)  
            background = (1.0 - a)[:, :, None] * torch.tensor([1.0, 1.0, 1.0], device=device) + a[:, :, None] * torch.tensor([0.5, 0.7, 1.0], device=device)
            colours = colours + throughput * torch.where(hitMask[..., None], torch.zeros_like(background), background)
            throughput = torch.where(hitMask[..., None], throughput, torch.zeros_like(throughput))
        albedo = torch.tensor([0.7, 0.7, 0.7], device=device) 
        randomTensor = torch.randn_like(closestNormals)
        randomTensor = torch.nn.functional.normalize(randomTensor, dim=-1)
        randomTensor = torch.where(torch.sum(randomTensor * closestNormals, dim=-1, keepdim=True) > 0,randomTensor, -randomTensor)
        origins = torch.where(hitMask[..., None], closestPoints + 1e-4 * closestNormals, origins)
        directions = randomTensor
        rays = directions  
        throughput = torch.where(hitMask[..., None], throughput * albedo, throughput)
    
    return colours