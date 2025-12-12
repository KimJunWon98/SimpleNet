import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "only-a",
    "only-b",
    "only-c",
    "only-d",
    "only-e",
    "only-f",
    "only-g",
    "only-h",
    "only-i",
    "only-j",
    "only-k",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def _is_image_file(fname: str) -> bool:
    return fname.lower().endswith(_IMAGE_EXTS)


class StitchingNetDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for StitchingNet (MVTec-like folder structure).
    """

    def __init__(
        self,
        source,
        classname,
        resize=224,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # mask가 없으면 올블랙(0) 마스크 반환
        if self.split == DatasetSplit.TEST and (mask_path is not None):
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskroot = os.path.join(self.source, classname, "ground_truth")

            if not os.path.isdir(classpath):
                raise FileNotFoundError(f"Split path not found: {classpath}")

            anomaly_types = [d for d in os.listdir(classpath) if os.path.isdir(os.path.join(classpath, d))]

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)

                # ✅ 이미지 파일만 필터링 (숨김파일/텍스트 등 제외)
                anomaly_files = sorted([f for f in os.listdir(anomaly_path) if _is_image_file(f)])
                img_list = [os.path.join(anomaly_path, x) for x in anomaly_files]
                imgpaths_per_class[classname][anomaly] = img_list

                # train/val split 처리
                if self.train_val_split < 1.0:
                    n_images = len(img_list)
                    split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = img_list[:split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = img_list[split_idx:]

                # ✅ TEST + anomaly(=good이 아님)일 때만 마스크 경로 구성
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_dir = os.path.join(maskroot, anomaly)

                    mask_list = []
                    if os.path.isdir(anomaly_mask_dir):
                        # 마스크도 이미지 파일만 필터링
                        mask_files = sorted([f for f in os.listdir(anomaly_mask_dir) if _is_image_file(f)])
                        mask_list = [os.path.join(anomaly_mask_dir, x) for x in mask_files]

                    # ✅ 핵심: 마스크가 없거나 / 개수가 안 맞으면 -> 이미지 개수만큼 None
                    n_imgs = len(imgpaths_per_class[classname][anomaly])
                    if len(mask_list) != n_imgs:
                        mask_list = [None] * n_imgs

                    maskpaths_per_class[classname][anomaly] = mask_list

                else:
                    # good이나 train/val에서는 mask 사용 안 함
                    maskpaths_per_class[classname][anomaly] = None

        # Unroll to iterate list
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                imgs = imgpaths_per_class[classname][anomaly]
                for i, image_path in enumerate(imgs):
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        mask_path = maskpaths_per_class[classname][anomaly][i]  # 항상 안전
                    else:
                        mask_path = None
                    data_to_iterate.append([classname, anomaly, image_path, mask_path])

        return imgpaths_per_class, data_to_iterate
