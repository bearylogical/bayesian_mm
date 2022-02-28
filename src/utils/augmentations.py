import albumentations as A

def default_aug():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.3),
        A.HorizontalFlip(),
        A.GaussianBlur(p=0.8),
        A.GaussNoise(p=0.4, var_limit=(0.02, 0.04)),
        A.RandomBrightnessContrast(p=0.5),

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))

