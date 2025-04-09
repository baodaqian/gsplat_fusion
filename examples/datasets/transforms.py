import os
import json
import numpy as np
import imageio.v2 as imageio
import torch

class TransformsParser:
    """
    A parser for NeRF-style JSON files containing camera transforms.
    This assumes your JSON file contains a "camera_angle_x" field and a "frames" list,
    where each frame has a "file_path" and a "transform_matrix".
    """
    def __init__(self, data_dir: str, json_filename: str = "transforms_train.json", factor: int = 1, test_every: int = 8):
        self.data_dir = data_dir
        self.factor = factor
        self.test_every = test_every

        json_path = os.path.join(data_dir, json_filename)
        assert os.path.exists(json_path), f"JSON file {json_path} does not exist."
        with open(json_path, "r") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]

        # List to store camera-to-world matrices and image paths.
        self.camtoworlds = []
        self.image_paths = []
        
        # Use the "file_path" in each frame. (Add appropriate extension if needed.)
        for frame in self.frames:
            # If your file_path doesn't include an extension, add one (e.g., ".png")
            file_path = frame["file_path"]
            if not os.path.splitext(file_path)[1]:
                file_path += ".png"
            full_path = os.path.join(data_dir, file_path)
            self.image_paths.append(full_path)
            # Use the provided 4x4 transform_matrix.
            self.camtoworlds.append(np.array(frame["transform_matrix"], dtype=np.float32))
            
        self.camtoworlds = np.stack(self.camtoworlds, axis=0)  # [num_images, 4, 4]
        
        # Assume all images have the same size. Read the first image.
        sample_img = imageio.imread(self.image_paths[0])
        self.height, self.width = sample_img.shape[:2]
        
        # Compute the intrinsic matrix from camera_angle_x.
        # A common formulation:
        #   focal_length = 0.5 * image_width / tan(0.5 * camera_angle_x)
        focal = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x)
        K = np.array([[focal, 0, self.width / 2.0],
                      [0, focal, self.height / 2.0],
                      [0, 0, 1]], dtype=np.float32)
        # If downsampling images, adjust intrinsics accordingly.
        if factor > 1:
            K[:2, :] /= factor
            self.width = int(round(self.width / factor))
            self.height = int(round(self.height / factor))
            
        self.K = K

        # Create a list that simply repeats K for every image.
        self.Ks = [K.copy() for _ in self.image_paths]

        # Optionally, you might want to define other properties (like dummy distortion parameters).
        # Here we add a simple dictionary to resemble the COLMAP parser.
        self.params_dict = {i: np.empty(0, dtype=np.float32) for i in range(len(self.image_paths))}
        # Use a simple index as the image id.
        self.camera_ids = list(range(len(self.image_paths)))

        # For consistency with the colmap parser, construct an "imsize_dict" mapping image index to (width, height)
        self.imsize_dict = {i: (self.width, self.height) for i in range(len(self.image_paths))}
        
        # Dummy values for points / points_rgb etc. (if you don't use depth)
        # If your pipeline doesn't need SFM point clouds, you can set these to empty arrays.
        self.points = np.empty((0, 3), dtype=np.float32)
        self.points_rgb = np.empty((0, 3), dtype=np.uint8)
        
        print(f"[TransformsParser] Loaded {len(self.image_paths)} images. Image size: {self.width} x {self.height}")

# For consistency with the Dataset, you can mimic the same interface.
class Dataset:
    """A simple dataset class that wraps the TransformsParser."""
    def __init__(self, parser: TransformsParser, split: str = "train", patch_size: int = None, load_depths: bool = False):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_paths))
        # You can use the test_every parameter to create a train/test split.
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        import cv2
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        # If you use a factor > 1, you might resize here.
        # For now, we assume the intrinsic matrix K already matches the image size.
        data = {
            "K": torch.from_numpy(self.parser.K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        return data
