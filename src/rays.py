
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
        closestMaterial = torch.full((*rays.shape[:2], 2), 0.0, device=device) 
        closestLight = torch.full((*rays.shape[:2], 3), 0.0, device=device)
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
            match obj.material.id:
                case 0:
                    closestMaterial = torch.where(mask.unsqueeze(-1), torch.tensor([0, 0.0], device=device).expand_as(closestMaterial), closestMaterial)
                case 1:
                    closestMaterial = torch.where(mask.unsqueeze(-1), torch.tensor([1, obj.material.fuzz], device=device).expand_as(closestMaterial), closestMaterial)
                case 2:
                    closestMaterial = torch.where(mask.unsqueeze(-1), torch.tensor([2, obj.material.emissive], device=device).expand_as(closestMaterial), closestMaterial)
                    closestLight = torch.where(mask.unsqueeze(-1), obj.colour, closestLight)
        hitMask = closest < float("inf")
        if not ~hitMask.any():
            a = 0.5 * (directions[:, :, 1] + 1)  
            background = (1.0 - a)[:, :, None] * torch.tensor([1.0, 1.0, 1.0], device=device) + a[:, :, None] * torch.tensor([0.5, 0.7, 1.0], device=device)
            colours = colours + throughput * torch.where(hitMask[..., None], torch.zeros_like(background), background)
            throughput = torch.where(hitMask.unsqueeze(-1), throughput, torch.zeros_like(throughput))
        albedo = closestColours
        randomTensor = torch.zeros_like(closestNormals)
        matType = closestMaterial[..., 0]
        diffuseMask = (matType == 0) & hitMask
        if diffuseMask.any():
            diffuse = torch.nn.functional.normalize(torch.randn_like(closestNormals), dim=-1)
            diffuse = diffuse + closestNormals
            randomTensor = torch.where(diffuseMask.unsqueeze(-1), diffuse, randomTensor)
        metalMask = (matType == 1) & hitMask
        if metalMask.any():
            metal = directions - (2* torch.sum((directions)*(closestNormals), dim=-1, keepdim=True) * closestNormals)
            randomTensor = torch.where(metalMask.unsqueeze(-1), metal, randomTensor)
        lightMask = (matType == 2) & hitMask
        if lightMask.any():
            emissiveness = torch.where(lightMask.unsqueeze(-1), closestMaterial[..., 1:2], torch.zeros_like(closestMaterial[..., 1:2])) 
            colours = colours + throughput * closestLight * emissiveness
            throughput = torch.where(lightMask.unsqueeze(-1), torch.zeros_like(throughput), throughput)
        origins = torch.where(hitMask.unsqueeze(-1), closestPoints + 1e-4 * closestNormals, origins)
        directions = randomTensor
        rays = directions  
        throughput = torch.where(hitMask.unsqueeze(-1), throughput * albedo, throughput)
    
    return colours