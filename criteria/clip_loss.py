import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import random
import random


def generate_bounding_boxes(img_width, img_height, min_patch_size, max_patch_size, batch_size):
    '''
    Function to generate bounding boxes for patches

    Args:
    img_width: Width of the original image
    img_height: Height of the original image
    min_patch_size: Minimum size of the patches to extract
    max_patch_size: Maximum size of the patches to extract
    batch_size: Number of bounding boxes to generate

    Returns:
    A list of tuples, each containing the top, left, bottom, and right coordinates of a bounding box
    '''
    bounding_boxes = []

    # Ensure that the patch sizes are smaller than the image dimensions
    if min_patch_size > min(img_width, img_height) or max_patch_size > min(img_width, img_height):
        raise ValueError("Patch sizes must be smaller than the image dimensions")

    for _ in range(batch_size):
        # Randomly select a size for the patch, ensuring it is smaller than both dimensions
        patch_size = random.randint(min_patch_size, min(max_patch_size, img_width, img_height))
        
        # Randomly select the top-left corner of the patch
        top = random.randint(0, img_height - patch_size)
        left = random.randint(0, img_width - patch_size)

        # Calculate the bottom-right corner of the patch
        bottom = top + patch_size
        right = left + patch_size

        # Add the bounding box to the list
        bounding_boxes.append((top, left, bottom, right))

    return bounding_boxes

def extract_patches(image, bbox_batch, resize = None):
    '''
    Function to extract patches from an image tensor

    Args:
    image: The original image tensor in CxHxW format
    bbox_batch: A batch of bounding boxes

    Returns:
    A list of image patch tensors
    '''
    patches = []

    for bbox in bbox_batch:
        top, left, bottom, right = bbox
        patch = image[:, top:bottom, left:right]
        if resize is not None:
            patch = resize(patch)
        patches.append(patch)

    return patches

imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

part_templates = [
    'the paw of a {}.',
    'the nose of a {}.',
    'the eye of the {}.',
    'the ears of a {}.',
    'an eye of a {}.',
    'the tongue of a {}.',
    'the fur of the {}.',
    'colorful {} fur.',
    'a snout of a {}.',
    'the teeth of the {}.',
    'the {}s fangs.',
    'a claw of the {}.',
    'the face of the {}',
    'a neck of a {}',
    'the head of the {}',
]

imagenet_templates_small = [
    'a photo of a {}.',
    'a rendering of a {}.',
    'a cropped photo of the {}.',
    'the photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a photo of my {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a photo of one {}.',
    'a close-up photo of the {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'a photo of the large {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
]

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):

    def __init__(self, direction_loss_type = 'cosine', distance_loss_type = 'mae', use_distance=False, src_img_list=None, tar_img_list=None):
        super(CLIPLoss, self).__init__()
        self.text_direction = None
        self.text_distance = None

        self.device = "cuda"
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.distance_loss = DirectionLoss(distance_loss_type)
        self.model, clip_preprocess = clip.load("ViT-B/16", device="cuda")
        self.preprocess = transforms.Compose(
                                             [transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC)] +  # to match CLIP input scale assumptions
                                             [clip_preprocess.transforms[-1]])  # + skip convert PIL to tensor
        # self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
        #                                       clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
        #                                       clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.use_distance = use_distance
        self.scale = 1.0

        if src_img_list and tar_img_list:
            self.feature_direction = None
            transform = transforms.ToTensor()
            src_features = 0
            for src_path in src_img_list:
                img = Image.open(src_path).convert('RGB')
                img = transform(img) # (3, h, w)
                img = img.unsqueeze(0) # (1, 3, h, w)
                img_feature = self.get_image_features(img)
                src_features += img_feature
            src_features /= len(src_img_list)

            tar_features = 0
            for tar_path in tar_img_list:
                img = Image.open(tar_path).convert('RGB')
                img = transform(img)  # (3, h, w)
                img = img.unsqueeze(0)  # (1, 3, h, w)
                img_feature = self.get_image_features(img)
                tar_features += img_feature
            src_features /= len(tar_img_list)

            self.src_features = src_features
            self.tar_features = tar_features

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def compute_text_direction(self, source_class: str, target_class: str, norm: bool = True) -> torch.Tensor:
        source_features = self.get_text_features(source_class, norm=norm)
        target_features = self.get_text_features(target_class, norm=norm)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        if norm:
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        if self.text_direction is None:
            self.text_direction = self.compute_text_direction(source_class, target_class)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        return self.direction_loss(edit_direction, self.text_direction).mean()

    def clip_distance_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        if self.text_distance is None:
            self.text_distance = self.compute_text_direction(source_class, target_class, norm=False) ** 2

            #print (self.text_distance)
        src_encoding = self.get_image_features(src_img, norm=False)
        target_encoding = self.get_image_features(target_img, norm=False)

        edit_distance = self.scale * (target_encoding - src_encoding) ** 2

        distance = (edit_distance - self.text_distance) ** 2
        print (distance.mean())
        return distance.mean()

        #return self.distance_loss(edit_distance, self.text_distance).mean()

    def compute_feature_direction(self):
        feature_direction = (self.tar_features - self.src_features).mean(axis=0, keepdim=True)
        feature_direction /= feature_direction.norm(dim=-1, keepdim=True)

        # norm = True
        feature_direction /= feature_direction.norm(dim=-1, keepdim=True)

        return feature_direction

    def clip_feature_directional_loss(self, src_img: torch.Tensor, target_img: torch.Tensor):
        if self.feature_direction is None:
            self.feature_direction = self.compute_feature_direction()

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        return self.direction_loss(edit_direction, self.feature_direction.detach()).mean()

    def forward_features(self, src_img: torch.Tensor, target_img: torch.Tensor):
        loss = self.clip_feature_directional_loss(src_img, target_img)
        return loss

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        loss = self.clip_directional_loss(src_img, source_class, target_img, target_class)

        if self.use_distance:
            loss_distance = 1.0 * self.clip_distance_loss(src_img, source_class, target_img, target_class)
            print("loss distance: ", loss_distance.item())
            loss = loss + loss_distance
        return loss
    
    def forward_patch_loss(self, src_img: torch.Tensor, source_class: str, tgt_img: torch.Tensor, target_class: str):
        if src_img.shape[0] != 3:
            src_img = src_img.squeeze(0)
        if tgt_img.shape[0] != 3:
            tgt_img = tgt_img.squeeze(0)
        h_, w_ = src_img.shape[-2], src_img.shape[-1] # assume src tgt same size
        bboxs = generate_bounding_boxes(w_, h_, 64, 224, 16)
        resize = transforms.Resize((224,224))
        src_patches, tgt_patches = extract_patches(src_img, bboxs, resize), extract_patches(tgt_img, bboxs, resize)
        src_patches, tgt_patches = torch.stack(src_patches), torch.stack(tgt_patches)
        if self.text_direction is None:
            self.text_direction = self.compute_text_direction(source_class, target_class)

        src_encoding = self.get_image_features(src_patches)
        target_encoding = self.get_image_features(tgt_patches)

        edit_direction = (target_encoding - src_encoding)
        # edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-5)

        text_direction = torch.stack([self.text_direction] * 16).squeeze(1)
        return self.direction_loss(text_direction,edit_direction).mean()

if __name__ == '__main__':
    # Example usage
    img_width = 256
    img_height = 256
    patch_size = 64
    batch_size = 4
    import PIL.Image as Image
    img = Image.open("/root/autodl-tmp/gaussian-splatting/data/fern/images/image001.png")
    import torch
    import numpy as np
    import torchvision
    img = torchvision.transforms.functional.to_tensor(img)
    # h, w = img.shape[1], img.shape[2]
    # bounding_boxes = generate_bounding_boxes(h, w, 64, 224, batch_size)
    # img_patches = extract_patches(img, bounding_boxes)
    # resize = transforms.Resize((64,64))
    # img_patches = [resize(_) for _ in img_patches]
    # img_patches = torch.stack(img_patches)

    # img_patches = resize(img_patches)

    # fig = torchvision.utils.make_grid(img_patches)
    # save fig
    from torchvision.utils import save_image
    # save_image(fig, 'patches.png')
    clip_loss = CLIPLoss()
    noise = torch.rand_like(img)* 0.1
    print(clip_loss.forward_patch_loss(img,"image",img+noise,"painting"))

    # print(bounding_boxes)