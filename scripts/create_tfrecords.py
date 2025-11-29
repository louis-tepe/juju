import os
import pandas as pd
import tensorflow as tf
import argparse
from tqdm import tqdm
import cv2
import sys

# Add project root to path to import src
sys.path.append(os.getcwd())

from src.data.preprocess import ben_graham_processing

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label, image_id):
    feature = {
        'image': _bytes_feature(image_string),
        'label': _int64_feature(label),
        'id': _bytes_feature(image_id.encode('utf-8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords(csv_path, image_dir, output_path, size=256):
    df = pd.read_csv(csv_path)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_id = row['id_code']
            label = int(row['diagnosis']) if 'diagnosis' in row else 0
            
            img_path = os.path.join(image_dir, f"{img_id}.png")
            
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found.")
                continue
            
            # Read and Preprocess
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply Ben Graham
            img = ben_graham_processing(img)
            
            # Resize
            img = cv2.resize(img, (size, size))
            
            # Encode to JPEG to save space in TFRecord
            img_string = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
            
            tf_example = image_example(img_string, label, img_id)
            writer.write(tf_example.SerializeToString())
            
    print(f"Created TFRecord at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train.csv")
    parser.add_argument("--dir", type=str, default="data/train_images")
    parser.add_argument("--output", type=str, default="data/processed/train.tfrec")
    parser.add_argument("--size", type=int, default=256)
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_tfrecords(args.csv, args.dir, args.output, args.size)
