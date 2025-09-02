import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import argparse

def create_dirs(output_dir):
    """
    Create the directory structure for the dataset.
    """
    outer_names = ['test', 'train']
    inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    os.makedirs(output_dir, exist_ok=True)
    for outer_name in outer_names:
        os.makedirs(os.path.join(output_dir, outer_name), exist_ok=True)
        for inner_name in inner_names:
            os.makedirs(os.path.join(output_dir, outer_name, inner_name), exist_ok=True)

def save_images(df, output_dir):
    """
    Save the images from the dataframe to the corresponding folders.
    """
    mat = np.zeros((48, 48), dtype=np.uint8)
    emotion_map = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'neutral'}
    usage_map = {'Training': 'train', 'PublicTest': 'test', 'PrivateTest': 'test'}
    counters = {usage: {emotion: 0 for emotion in emotion_map.values()} for usage in ['train', 'test']}

    for i in tqdm(range(len(df))):
        txt = df['pixels'][i]
        words = txt.split()

        for j in range(2304):
            xind = j // 48
            yind = j % 48
            mat[xind][yind] = int(words[j])

        img = Image.fromarray(mat)
        usage = usage_map[df['Usage'][i]]
        emotion = emotion_map[df['emotion'][i]]
        count = counters[usage][emotion]
        img.save(os.path.join(output_dir, usage, emotion, f'im{count}.png'))
        counters[usage][emotion] += 1

def main():
    parser = argparse.ArgumentParser(description='Prepare the FER2013 dataset.')
    parser.add_argument('--csv', default='../fer2013.csv', help='Path to the fer2013.csv file.')
    parser.add_argument('--output', default='../data', help='Path to the output directory.')
    args = parser.parse_args()

    create_dirs(args.output)
    df = pd.read_csv(args.csv)
    save_images(df, args.output)
    print("Done!")

if __name__ == '__main__':
    main()
