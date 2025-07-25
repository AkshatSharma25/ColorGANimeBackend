import cv2
import cv2
import numpy as np
from torchvision import transforms
import os
import json
from collections import defaultdict
import numpy as np
from PIL import Image, ImageEnhance
import glob
import os
import json
import generator
import torch
from torchvision.utils import save_image






def breakIntoPieces(input_dir = r"uploads",output_dir = r"fragments",metadata_file = "fragment_metadata.json"):

    os.makedirs(output_dir, exist_ok=True)

    metadata_list = []
    panel_num = 0
    # print(os.listdir())
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    # print(image_paths)
    # image_paths.append("\\uploads\\page.png")
    for img_path in image_paths:
        print(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 100:
                fragment = img[y:y+h, x:x+w]
                resized_fragment = cv2.resize(fragment, (128, 128), interpolation=cv2.INTER_AREA)
                fragment_name = f"fragment{panel_num}.png"
                fragment_path = os.path.join(output_dir, fragment_name)
                # print(fragment_path)
                cv2.imwrite(fragment_path, resized_fragment)
                metadata_list.append({
                    "base_image": os.path.basename(img_path),
                    "original_width": img.shape[1],
                    "original_height": img.shape[0],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "fragment_filename": fragment_name
                })
                panel_num += 1

    with open(metadata_file, "w") as f:
        json.dump(metadata_list, f, indent=2)




def colorise(gen=generator.gen,DEVICE=generator.DEVICE,input_dir= r'fragments'  ,output_folder = 'uploads'):

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])  # maps [0,1] → [-1,1]
    ])


    target_size = (128, 128)
    gen.eval()
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    # print(image_paths)
    i=0
    for image in image_paths:
        # print(image)
        img = Image.open(image)
        # save_image(img,f"C:\\\\code library\\\\ColorGANime\\\\ColorGANime2\\\\output_folder\\\\input{i}.png")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE) 
        with torch.no_grad():
            output = gen(input_tensor)  # Output shape: [1, 3, 128, 128]
            # print(image.split())
            name=image.split('\\');
            output = (output + 1) / 2
            save_image(output, f"fragments\\{name[-1]}")  
            enhance_to_match(f"fragments\\{name[-1]}",f"fragments\\{name[-1]}")





# input_dir=r"C:\\code library\\ColorGANime\\ColorGANime2\\output_fragments"
def enhance_to_match(input_img_path, out_path, 
                    contrast_factor=1.1, 
                    saturation_factor=1.3, 
                    sharpness_factor=1.7, 
                    brightness_factor=1.0):
    img = Image.open(input_img_path).convert("RGB")
    
    # Adjust enhancements as necessary
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Color(img).enhance(saturation_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    
    img.save(out_path)





def reconstruct(fragment_dir = r"fragments",output_dir = r"\uploads"):
    fragment_dir = r"fragments"
    output_dir = r"uploads"
    metadata_file = "fragment_metadata.json"
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata_list = json.load(f)

    # Group fragments by base image
    fragments_by_image = defaultdict(list)
    for item in metadata_list:
        fragments_by_image[item["base_image"]].append(item)

    for base_image, fragments in fragments_by_image.items():
        # Assume all fragments have same original dimensions
        original_width = fragments[0]['original_width']
        original_height = fragments[0]['original_height']
        canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        # np.fill(canvas,1)
        for i in range(original_height):
            for j in range(original_width):
                for k in range(3):
                    canvas[i][j][k]=255;
        for frag in fragments:
            frag_path = os.path.join(fragment_dir, frag["fragment_filename"])
            fragment_img = cv2.imread(frag_path)
            if fragment_img is None:
                continue
            # Resize fragment to the original bounding box size
            fragment_resized = cv2.resize(fragment_img, (frag["w"], frag["h"]))
            x, y, w, h = frag["x"], frag["y"], frag["w"], frag["h"]
            canvas[y:y+h, x:x+w] = fragment_resized
        save_path = os.path.join(output_dir, f"reconstructed_{base_image}")
        # print(save_path)
        # cv2.imwrite(save_path, canvas)
        cv2.imwrite(r"return.png",canvas)

def delete_all_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
