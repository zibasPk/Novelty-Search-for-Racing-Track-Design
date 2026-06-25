import json
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
import xml.etree.ElementTree as ET
from enum import Enum
import os
import sys



cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")

# Define the path to the 'qd' folder
# We assume the notebook is running from the root 'Quality-Diversity-...' folder
qd_path = os.path.join(cwd, 'qd')

# Add it to the system path so Python can find config.py, utils.py, etc.
if qd_path not in sys.path:
    sys.path.append(qd_path)
    print(f"Added '{qd_path}' to sys.path")


SegmentType = Enum('SegmentType', [('str',0), ('lft',-1), ('rgt',1)])

AVERAGE_LENGTH = 5  # segments

def create_dataset(xml_source_folder, json_source_folder, output_file, max_files=None):
    """
    Create a dataset from XML track files, only including tracks that have corresponding JSON files in the specified folder.
    The dataset consists of padded track segment arrays and corresponding masks.
    """
    dataset = []
    ids = []
    
    tracks_skipped = 0
    max_track_len = 0
    json_source_path = Path(json_source_folder)
    json_files = list(json_source_path.glob("*.json"))
    

    for index, file_path in enumerate(json_files):  
        if max_files is not None and index >= max_files:
            break
        
        # read the json file and extract the id
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            track_id = data.get("id")
            if track_id is None:
                print(f"Warning: Missing track ID in {file_path.name}, skipping this file.")
                tracks_skipped += 1
                continue
        
        xml_file_path = Path(xml_source_folder) / f"output_{track_id}.xml"
        if not xml_file_path.exists():
            print(f"Warning: XML file for track ID {track_id} not found, skipping this file.")
            tracks_skipped += 1
            continue
        
        with open(xml_file_path, "r") as f:
            xml_data = f.read()
        # Replace undefined entity with a real value
        xml_data = xml_data.replace("&default-surfaces;", "SURFACE_VALUE")
        xml_data = xml_data.replace("&default-objects;", "OBJECT_VALUE")
        
        if (index+1) % 1000 == 0:
            print (f"Processing the {index+1}th file: {file_path.name}")
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            print(f"Failed to parse {file_path.name}: {e}")
        
        segments = root.findall(
            ".//section[@name='Main Track']/section[@name='Track Segments']/section"
        )
        
        track = []
        for s in segments:
            type = s.find(".//attstr[@name='type']")
            type_val = type.attrib.get("val")
            
            type_enum = SegmentType[type_val]
            
            if type_val == "str":
                length = s.find(".//attnum[@name='lg']")
                length_val = float(length.attrib.get("val"))
                if (length_val <= 0):
                    print(f"Warning: Non-positive length in {file_path.name} with val {length_val}, skipping segment.")
                    continue
                #split straight segments longer than AVERAGE_LENGTH
                if length_val > AVERAGE_LENGTH:
                    num_splits = int(np.ceil(length_val / AVERAGE_LENGTH))
                    split_length = length_val / num_splits
                    for _ in range(num_splits):
                        track.append([split_length, 0])
                else:
                    track.append([length_val, 0])
            elif type_val =="lft" or type_val =="rgt":
                arc = s.find(".//attnum[@name='arc']")
                radius = s.find(".//attnum[@name='radius']")
                arc_rad = float(arc.attrib.get("val")) * (np.pi / 180)
                radius_val = float(radius.attrib.get("val"))
                length_val = arc_rad * radius_val
                track.append([length_val, arc_rad * type_enum.value])
                
            else:
                print(f"Unknown segment type in {file_path.name}, skipping segment.")
        
        if len(track) > max_track_len:
            max_track_len = len(track)
        if len(track) <= 2:
            tracks_skipped += 1
            continue  # skip empty tracks
        dataset.append(np.array(track))
        ids.append(track_id)  
    if not dataset:
        print("No data collected. Exiting.")
        return
  
    # flatten the dataset and create an index array
    flattened_dataset = np.concatenate(dataset, axis=0)
    lengths = np.array([len(track) for track in dataset])
    index_array = np.cumsum(lengths[:-1])
        
 
    ids = np.array(ids)
    print ("longest track length (in segments): ", max_track_len)
    print (f"Total tracks skipped due to insufficient length: {tracks_skipped}")
    np.savez_compressed(output_file, data=flattened_dataset, indices=index_array, ids=ids)


if __name__ == "__main__":
    XML_SOURCE_DIR = 'data/voronoi/xmlTracks'
    JSON_SOURCE_DIR = 'data/voronoi/fitted'
    # Changed extension to .npz
    OUTPUT_NAME = 'data/datasets/xml/dataset10k.npz'

    # Ensure output directory exists
    Path('data/datasets/xml').mkdir(parents=True, exist_ok=True)

    create_dataset(XML_SOURCE_DIR,JSON_SOURCE_DIR, OUTPUT_NAME, 10000)
