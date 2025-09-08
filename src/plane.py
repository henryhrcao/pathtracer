from object import *
class Plane(Object):
    def __init__(self, points, colour, material):
        self.points = points
        super().__init__(colour, material)
    def intersect(self, rays, origins):
        directions  = torch.nn.functional.normalize(rays, dim=-1)
        normal = torch.linalg.cross((self.points[2]-self.points[1]),((self.points[0]-self.points[1])))
        normal = torch.nn.functional.normalize(normal, dim=-1)
        if torch.sum(normal*self.points[0], dim=-1) != 0:
            t = (torch.sum(normal*self.points[0], dim=-1) - torch.sum(normal*origins, dim=-1)) / torch.sum(normal*directions, dim=-1)
            intersectPoints = origins + directions * t.unsqueeze(-1)
            tOne = torch.sum(normal * torch.linalg.cross((self.points[1]-self.points[0]).unsqueeze(0).expand_as(intersectPoints),((intersectPoints-self.points[0]))), dim=-1)
            tTwo = torch.sum(normal * torch.linalg.cross((self.points[2]-self.points[1]).unsqueeze(0).expand_as(intersectPoints),((intersectPoints-self.points[1]))), dim=-1)
            tThree = torch.sum(normal * torch.linalg.cross((self.points[3]-self.points[2]).unsqueeze(0).expand_as(intersectPoints),((intersectPoints-self.points[2]))), dim=-1)
            tFour = torch.sum(normal * torch.linalg.cross((self.points[0]-self.points[3]).unsqueeze(0).expand_as(intersectPoints),((intersectPoints-self.points[3]))), dim=-1)
            mask = (tOne >= 0) & (tTwo >= 0) & (tThree >= 0) & (tFour >= 0)
            return t, intersectPoints, normal, mask


