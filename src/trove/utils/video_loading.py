import numpy as np
import cv2

# Reference: https://github.com/OpenGVLab/InternVideo/blob/09d872e5093296c6f36b8b3a91fc511b76433bf7/Data/InternVid/viclip/__init__.py#L27
def _frame_from_video(video):
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    return frames

def load_frame_from_video(df, image_idx, data_dir): 
    image_idx_to_sequence_idx = dict(zip(
        np.arange(df['image_probs'].explode().shape[0]), 
        df['image_probs'].explode().index            
    ))
    image_counter = np.concatenate(
        df['image_probs'].apply(lambda x: np.arange(len(x)))
    )
    num_images_per_seq = df["image_probs"].apply(lambda x: len(x)).values

    sequence_idx = image_idx_to_sequence_idx[image_idx]
    video_fp = df.iloc[sequence_idx]['filepath']
    image_idx_in_sequence = image_counter[image_idx]

    video = cv2.VideoCapture(f"{data_dir}/kinetics400/{video_fp}")
    images = _frame_from_video(video)
    
    step = len(images) // num_images_per_seq[sequence_idx]
    images = images[::step][:num_images_per_seq[sequence_idx]]
    images = [cv2.resize(x[:, :, ::-1], (224, 224)) for x in images]
    
    return images[image_idx_in_sequence]
