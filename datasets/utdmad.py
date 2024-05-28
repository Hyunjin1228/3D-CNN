import os
import glob
import tarfile
from tqdm import tqdm
import pickle
import re
from torchvision import transforms, io
import torch
import sys
import fire

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(root_dir, 'models'))

from model import hardwire_layer, auxiliary_feature

CATEGORIES = [
    "wave",
    "clap",
    "jog",
    "walk",
    "standtosit",
    "sittostand"
]

base_transform = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.Grayscale(num_output_channels=1)
])

def make_utdmad_dataset(directory="utdmad-data-aux", transform=None, f=9, device=None):
    paths = []
    set = ['3', '4', '22', '23', '24', '25']
    for filename in glob.glob('./datasets/utdmad/RGB/*.avi'):
        for i in set:
            if 'a'+i in filename:
#                print(f)
                paths.append(filename)
                
    if not transform :
        transform = base_transform

    print("Processing ...")
    # print(os.getcwd())
    dir_path = os.path.join(os.getcwd(),"datasets", directory) # directory path that the processed dataset will be stored
    subjects = 8
    filenames = [[] for _ in range(subjects)] 
    category_idx = 10
    subject_idx = 0
    for filename in paths:
        for i in range(subjects):
            if 's' + str(i+1) in filename:
                subject_idx = i
        #print(category_idx, subject_idx, filename)
        filenames[subject_idx].append(filename)
            
    for subject_id in range(subjects):
        categories = []
        input = []
        
        for filename in tqdm(filenames[subject_id], desc='subject ' + str(subject_id)):
            # Get category in this video.
            category = 0
            for cnt, i in enumerate(set):
                if 'a'+i in filename:
                    category = CATEGORIES[cnt]
            # print(category, filename)
            file_path = filename

            frames = io.read_video(file_path, output_format='TCHW')[0] / 255.0
            frames = transform(frames).split(1, dim=0)
            seg_frames = torch.cat(frames, dim = 0)
            N, throw = seg_frames.shape[0] // f, seg_frames.shape[0] % f
            if throw > 0:
                    seg_frames = seg_frames[:-throw]
            seg_frames = seg_frames.reshape(N, f, seg_frames.shape[-2], seg_frames.shape[-1])
            categories.extend([category for _ in range(N)])
            # input.append(seg_frames)
            input = seg_frames
            # input = torch.cat(seg_frames, dim=0)
            input = torch.tensor(input)

        # hardwiring layer
        input = hardwire_layer(input, device, verbose=True).cpu() # Tensor shape : (N, f, h, w) -> (N, 1, 5f-2, h, w)
        gray_img = input[:,:,:f, :, :]
        input_aux = auxiliary_feature(gray_img)

        # save the data per each subject
        person_path = os.path.join(dir_path, str(subject_id))
        # print(categories, subject_id)
        
        data = {
            "category": categories,
            "input": input, # Tensor shape : (N, 1, 5f-2, h, w)
            "subject": subject_id,
            "aux": input_aux, # Tensor shape : (N, 30)
        }

        person_path = os.path.join(dir_path, str(subject_id))
        pickle.dump(data, open("%s.p" % person_path, "wb"))
        


if __name__ == "__main__":
    #TODO : add kth_download
    print("Making utd-mad dataset")
    fire.Fire(make_utdmad_dataset())



