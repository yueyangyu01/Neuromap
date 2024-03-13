### imports
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    Transform,
    ToTensor
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

import numpy as np
import boto3
import json
import io

def list_image_paths(patient_id, bucket_name='neuromapuserimages'):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=patient_id)
    image_paths = []

    for item in response.get('Contents', []):
        image_key = item['Key']
        image_path = f"s3://{bucket_name}/{image_key}"
        image_paths.append(image_path)

    return image_paths

def transform_data(input_path):
    s3 = boto3.client('s3')
    imgs_data = []
    for s3_path in input_path:
        # Parse the S3 path to get the bucket name and object key
        bucket_name = s3_path.split('/')[2]
        object_key = '/'.join(s3_path.split('/')[3:])

        # Fetch the object from S3
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)

        # Read the object data into a BytesIO buffer
        buffer = io.BytesIO(obj['Body'].read())
        img = nib.load(buffer)
        img_data = img.get_fdata()
        imgs_data.append(img_data)

    # Makes a 4D array
    image_data = np.stack(imgs_data, axis=0)

    # 4D nibabel image as image
    image_dict = {"image": image_data}

    transforms = Compose(
        [
            ToTensor(),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    transformed_imgs = transforms(image_dict)

    final_img = transformed_imgs['image']

    return final_img


def model_fn(model_path):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

    # Put model in eval mode for inference
    model.eval()

    return model

def inference(model, input, VAL_AMP=False):
    def _compute(input_img):
        return sliding_window_inference(
            inputs=input_img,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def output_fn(prediction, accept='application/json'):
    if type(prediction) == np.ndarray:
        pred_dims = prediction.shape
        pred_bytes = prediction.tobytes()
        pred_type = str(np.dtype(prediction))
        prediction = {
            'shape': pred_dims,
            'data': pred_bytes,
            'dtype': pred_type,
        }
    if accept == 'application/json':
        return json.dumps({'Body': prediction})