import numpy as np

class CheckDetection:
    def __init__(self, cam):

        if cam == 1:
            #start, end = sv.Point(x=-500, y=292), sv.Point(x=1878, y=292)
            self.attention_vector1 = [[0,175],[1279,175]], ">"
            self.attention_vector2 = [[0,505],[1140,0]], ">"
        elif cam == 2:
            #start, end = sv.Point(x=-500, y=711), sv.Point(x=1878, y=198)
            self.attention_vector1 = [[0,120],[1279,570]], ">"
            self.attention_vector2 = [[63,0],[412,960]], "<"
        elif cam == 3:
            #start, end = sv.Point(x=-500, y=600), sv.Point(x=1278, y=300)
            self.attention_vector1 = [[0,100],[2086,400]], ">"
            self.attention_vector2 = [[1500,0],[1900,2000]], ">"
        elif cam == 5:
            #start, end = sv.Point(x=1600, y=200), sv.Point(x=2600, y=2500)
            self.attention_vector1 = [[0,3000],[2000,0]], ">"
            self.attention_vector2 = None

    def verify_attention(self, bbox):
        
        center_point = self.get_center(bbox)

        attention = self.is_point_in_attention(center_point)

        return attention



    def get_center(self,bbox):
        
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2])
        
        return bbox_center
    
    def is_point_in_attention(self,point):

        vector1 = self.attention_vector1
        vector2 = self.attention_vector2
        
        if vector1 is not None and vector2 is not None:
            # Assuming vector1 and vector2 are represented by [x, y] points
            vector1, sign1 = vector1
            vector2, sign2 = vector2
            v1p1, v1p2 = vector1
            v2p1, v2p2 = vector2
            # Calculate the cross product to determine if the point is on the same side of the line
            if(sign1 == ">"):
                cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
            else:
                cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
            if(sign2 == ">"):
                cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) > 0
            else:
                cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) < 0
        elif(vector1 is not None):
            vector1, sign1 = vector1
            v1p1, v1p2 = vector1
            if(sign1 == ">"):
                cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
            else:
                cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
            cross_product2 = True
        else:
            cross_product = True
            cross_product2 = True

        cross_product = cross_product and cross_product2

        # If the cross product is positive, the point is on the same side as the frame
        return cross_product
        


        

