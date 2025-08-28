import torch
import sphere
def colour(rays,device,origin, objects):
    directions = torch.nn.functional.normalize(rays, dim=-1)
    closest = torch.full(rays.shape[:2], float("inf"), device=device)
    closestNormals = torch.full((*rays.shape[:2], 3), 0.0, device=device)
    for object in objects:
        roots = object.intersect(rays, origin)
        points = origin + directions * roots.unsqueeze(-1)
        normals = torch.nn.functional.normalize(points - object.center, dim=-1)
        mask = roots < closest
        closest = torch.where(mask , roots, closest)
        closestNormals = torch.where(mask.unsqueeze(-1), normals, closestNormals)
    closest = (closest < float("inf"))[...,None]
    a = 0.5*(directions[:,:,1]+1)
    backroundColours = (1.0-a)[:,:,None]*torch.tensor([1,1,1],device= device) + a[:,:,None]*torch.tensor([0.5,0.7,1],device= device)
    colours = torch.where(closest,0.5 * closestNormals + 0.5,backroundColours)
    return colours