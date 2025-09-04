from object import *
class Sphere(Object):
    def __init__(self, center, radius, colour, material):
        self.center = center
        self.radius = radius
        super().__init__(colour, material)
    def intersect(self, rays, origins):
        directions  = torch.nn.functional.normalize(rays, dim=-1)
        oc = origins - self.center.view(1, 1, 3)
        quadB = 2 * torch.sum(directions*(oc), dim=-1)
        quadA = torch.sum(directions*directions, dim=-1)
        quadC = torch.sum((oc)*(oc), dim=-1) - (self.radius*self.radius)
        discriminants = (quadB*quadB) - (4 * quadA * quadC)
        sqrtDisc = torch.sqrt(discriminants.clamp(min=0))
        t = torch.full_like(discriminants, float("inf"))
        root1 = (-quadB - sqrtDisc) / (2*quadA)
        root2 = (-quadB + sqrtDisc) / (2*quadA)
        mask = discriminants >= 0.001
        minroot = torch.minimum(root1,root2)
        minroot = torch.where(minroot>0,minroot,torch.maximum(root1,root2))
        roots = torch.where(mask & (minroot>0),minroot,t)
        points = origins + directions * roots.unsqueeze(-1)
        normals = torch.nn.functional.normalize(points - self.center, dim=-1)
        return roots, points, normals
