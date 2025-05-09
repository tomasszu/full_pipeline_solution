from ReceiveDetections import ReceiveDetectionsService
from ExtractingFeatures import ExtractingFeatures
from SendFeatures import SendFeatures
from CheckDetection import CheckDetection

# In your main loop (e.g., every N ms)
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['save', 'comp'], default='save')
    return parser.parse_args()

def main(args):

    receiver = ReceiveDetectionsService()
    extractor = ExtractingFeatures()
    check = CheckDetection(1)

    if(args.mode == 'comp'):
        sender = SendFeatures(mqtt_topic="tomass/compare_features")
    else:
        sender = SendFeatures(mqtt_topic="tomass/save_features")


    while True:
        new_images = receiver.get_pending_images()
        for entry in new_images:

            image = entry["image"]
            track_id = entry["track_id"]
            bbox = entry["bbox"]
            #print(f"Ready for feature extraction: Track {track_id}, Shape: {image.shape}")

            if check.perform_checks(track_id, bbox):

                features = extractor.get_feature(image)

                sender(track_id, features)
            
        #time.sleep(0.1)


if __name__ == "__main__":
    args = parse_args()
    main(args)

