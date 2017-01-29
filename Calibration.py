class Calibration:
    def __init__(self, mtx=0, dist=0):
        # Camera calibraton matrix
        self.mtx = mtx
        # Camera distortion coefficients
        self.dist = dist