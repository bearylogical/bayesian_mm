import albumentations as A

def default_aug():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.3),
        A.OneOf([
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5, var_limit=(0.002, 0.003)),
        ], p=0.3),
        A.RandomBrightnessContrast(p=0.5, contrast_limit=0.5, brightness_limit=0.1),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))

# def field_of_view():
#     return A.Compose([
#         A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
#     ],keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))