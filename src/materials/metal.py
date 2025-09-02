from materials.material import *
class Metal(Material):
    def __init__(self, fuzziness):
        self.fuzz = fuzziness
        super().__init__(id=1)