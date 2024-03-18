from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import math

def resize_and_pad(img, target_width=224, fill=(0, 0, 0)):
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    new_width = target_width
    new_height = int(new_width * aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Check if padding is needed (in case the new height is less than the target size)
    target_size = (target_width, target_width)  # Define target size if you want to maintain a square image after resizing
    pad_vertical = max(0, target_size[1] - new_height) // 2
    pad_horizontal = (target_size[0] - new_width) // 2  # Should be zero in this case but included for completeness
    
    # Apply padding if needed
    img_padded = Image.new('RGB', target_size, fill)
    img_padded.paste(img_resized, (pad_horizontal, pad_vertical))
    
    return img_padded

def resize(img, target_width=512):
    original_width, original_height = img.size
    aspect_ratio = original_height / original_width
    new_width = target_width
    new_height = int(new_width * aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img_resized

def split_and_stack(img, target_width=224):
    orig_w, orig_h = img.size
    
    # Step 1: Calculate 'x'
    aspect_ratio = orig_h / orig_w
    x = math.ceil(orig_w / orig_h)
    
    # Step 2: Split the image into 'x' vertical patches
    patch_width = orig_w // x  # Each patch's width
    
    patches = []
    for i in range(x):
        left = i * patch_width
        right = left + patch_width if i < x - 1 else orig_w  # Adjust the last patch's width if necessary
        patch = img.crop((left, 0, right, orig_h))
        patches.append(patch)
    
    # Step 3: Scale patches to have a width of 224px, maintaining aspect ratio
    scaled_patches = [patch.resize((target_width, int(target_width*aspect_ratio)), Image.Resampling.LANCZOS) for patch in patches]
    
    # Step 4: Stack scaled patches vertically
    total_height = sum(patch.height for patch in scaled_patches)
    final_img = Image.new('RGB', (target_width, total_height))
    
    y_offset = 0
    for patch in scaled_patches:
        final_img.paste(patch, (0, y_offset))
        y_offset += patch.height

    final_img = final_img.resize((target_width, target_width), Image.Resampling.LANCZOS)
    
    return final_img

class CarMovementDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        """
        Args:
            pairs (list of tuples): List of tuples containing image paths for the pairs.
            labels (list): List of labels indicating movement (1) or no movement (0).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2 = self.pairs[idx]
        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)
        label = self.labels[idx]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Optionally, you can concatenate or stack images here if your model expects a single input tensor
        return (image1, image2), label