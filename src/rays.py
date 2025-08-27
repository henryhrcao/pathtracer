import torch
import sphere
def colour(rays,device,origin, objects):
    normalized = torch.nn.functional.normalize(rays, dim=-1)
    closest= torch.full(rays.shape[:2], float("inf"), device=device)
    #intersects = objects[0].intersect(rays, origin)
    #intersects = (intersects < float("inf"))[...,None]
    for object in objects:
        roots = object.intersect(rays, origin)
        mask = roots < closest
        closest = torch.where(mask , roots, closest)
    closest = (closest < float("inf"))[...,None]
    red = torch.tensor([1,0,0], device=device)
    a = 0.5*(normalized[:,:,1]+1)
    backroundColours = (1.0-a)[:,:,None]*torch.tensor([1,1,1],device= device) + a[:,:,None]*torch.tensor([0.5,0.7,1],device= device)
    colours = torch.where(closest,red,backroundColours)
    return colours
def sphereIntersect(center, rays, origin, radius):
    directions  = torch.nn.functional.normalize(rays, dim=-1)
    oc = (origin - center).view(1,1,3)  
    quadB = 2 * torch.sum(directions*(oc), dim=-1)
    quadA = torch.sum(directions*directions, dim=-1)
    quadC = torch.sum((oc)*(oc), dim=-1) - (radius*radius)
    discriminants = (quadB*quadB) - (4 * quadA * quadC)
    sqrtDisc = torch.sqrt(discriminants.clamp(min=0))
    t = torch.full_like(discriminants, float("inf"))
    root1 = (-quadB - sqrtDisc) / (2*quadA)
    root2 = (-quadB + sqrtDisc) / (2*quadA)
    mask = discriminants >= 0
    minroot = torch.minimum(root1,root2)
    minroot = torch.where(minroot>0,minroot,torch.maximum(root1,root2))
    roots = torch.where(mask & (minroot>0),minroot,t)
    return roots