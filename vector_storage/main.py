from ReceiveFeatures import ReceiveFeatures
from database import Database

import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['save', 'comp'], default='save')
    return parser.parse_args()

def main(args):

    db = Database()

    if(args.mode == 'comp'):
        receiver = ReceiveFeatures(topic="tomass/compare_features")

        while True:
            comp_vectors = receiver.get_pending_vectors()

            for entry in comp_vectors:

                id = entry["track_id"]
                vector = entry["features"]

                print(id, " found as: ", db.query(vector)[0])
                
            #time.sleep(0.01)
        
    else:
        receiver = ReceiveFeatures(topic="tomass/save_features")

        while True:
            new_vectors = receiver.get_pending_vectors()

            for entry in new_vectors:

                id = entry["track_id"]
                vector = entry["features"]

                db.insert(id, vector)
                
            #time.sleep(0.01)

    




if __name__ == "__main__":
    args = parse_args()
    main(args)