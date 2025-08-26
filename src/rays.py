import torch
def colour(rays,device):
    normalized = torch.nn.functional.normalize(rays, dim=-1)
    a = 0.5*(normalized[:,:,1]+1)
    return (1.0-a)[:,:,None]*torch.tensor([1,1,1],device= device) + a[:,:,None]*torch.tensor([0.5,0.7,1],device= device)
def sphereIntersect(rays, origin, radius, device):
    
    pass