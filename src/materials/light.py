from materials.material import *
class Light(Material):
    def __init__(self, emissive):
        self.emissive = emissive
        super().__init__(id=2)