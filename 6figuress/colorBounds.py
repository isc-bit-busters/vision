import numpy as np  

class ColorBound:
    def __init__(self): # Initialize the min and max values for each HSV channel
        # self.min_black = np.array([85, 0, 0]) 
        # self.max_black = np.array([179, 255, 90]) 
        #self.min_black = np.array([0, 0, 0]) 
        # self.max_black = np.array([179, 110, 85]) 
        # # [179, 130, 80]
        self.min_white = np.array([0, 0, 200])
        self.max_white = np.array([180, 50, 255])

