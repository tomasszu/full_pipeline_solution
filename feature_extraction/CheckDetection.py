import numpy as np

class CheckDetection:
    def __init__(self, cam):

        if cam == 1:
            #start, end = sv.Point(x=-500, y=292), sv.Point(x=1878, y=292)
            self.attention_vector1 = [[0,175],[1279,175]], ">"
            self.attention_vector2 = [[0,505],[1140,0]], ">"
            #------------Crop Zone definitions for EDI camera 1------------------.
            zRows, zCols = 9, 5
            frame_height = 960
            frame_width = 1280
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

        self.zones = []
        self.zone_of_detections = {}

        # Calculate the width and height of each count zone based on the frame dimensions
        zone_width = frame_width // zCols
        zone_height = frame_height // zRows

        # Create the zones (rectangles) and store their coordinates
        for i in range(zRows):
            for j in range(zCols):
                x1 = j * zone_width
                y1 = i * zone_height
                x2 = (j + 1) * zone_width
                y2 = (i + 1) * zone_height
                if(y1 > 250 and x1 > 75 and x2 < 1200 and y1 < 700):
                    self.zones.append((x1, y1, x2, y2))



    def perform_checks(self, track_id, bbox):
        # Perform checks on the bounding box

        # If the bounding box is in the attention area, return True
        if self.verify_attention(bbox):
            if self.check_crop_zones(track_id, bbox):
                # If the bounding box is in a crop zone, return True
                return True
        # If the bounding box is not in the attention area or crop zone, return False
        return False

    def verify_attention(self, bbox):
        
        center_point = self.get_center(bbox)

        attention = self.is_point_in_attention(center_point)

        return attention
    
    def check_crop_zones(self, track_id, bbox):

        center_point = self.get_center(bbox)

        zone = self.zone_of_point(center_point)
        if zone != -1:
            # If the point is in a zone
            if((track_id not in self.zone_of_detections) or (self.zone_of_detections[track_id] != zone)):
                # If the track_id is not in the dictionary or the zone has changed
                # Update the zone of detections for the track_id
                self.zone_of_detections.update({track_id: zone})
                return True
        else:
            # If the point is not in any zone, remove the track_id from the dictionary
            if track_id in self.zone_of_detections:
                del self.zone_of_detections[track_id]
                return False
        # If the point is not in any zone, return False
            return False


    def zone_of_point(self, point):
        """
        Determine the zone index in which a given point lies.
        
        Args:
        - point (tuple): (x, y) coordinates of the point
        - zones (list): List of zone coordinates, each zone represented as ((x1, y1), (x2, y2))
        
        Returns:
        - zone_index (int): Index of the zone in which the point lies, or -1 if it's not in any zone
        """
        x, y = point
        
        # Iterate through each zone and check if the point lies within it
        for zone_index, zone in enumerate(self.zones):
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_index
        
        # If the point is not in any zone, return -1
        return -1




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
        


        

