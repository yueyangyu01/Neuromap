### imports
import matplotlib.pyplot as plt
import nibabel as nib
import torch
from monai.transforms import (
    Compose,

    NormalizeIntensityd,
    Orientationd,
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
import os
import tempfile

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
    img_headers = []
    for s3_path in input_path:
        # Parse the S3 path to get the bucket name and object key
        bucket_name = s3_path.split('/')[2]
        object_key = '/'.join(s3_path.split('/')[3:])

        # This will run on Sagemaker, try and see if it works
        if 'nii.gz' in object_key :
            img = nib.load(s3_path)
            img_data = img.get_fdata()
            imgs_data.append(img_data)
            img_headers.append(img.header)

    # Makes a 4D array
    image_data = np.stack(imgs_data, axis=0)

    # 4D nibabel image as image
    image_dict = {"image": image_data}

    transforms = Compose(
        [
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)      
        ]
    )

    transformed_imgs = transforms(image_dict)

    print("Image is transformed")
    final_img = transformed_imgs['image']

    return final_img, img_headers



def model_fn(model_path):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    model_path = os.path.join(model_path, 'model.pth')

    try:
        model.load_state_dict(torch.load(model_path))
    except:
        with open(model_path, 'rb') as f:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(f))
            else:
                model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

    # Put model in eval mode for inference
    model.eval()

    return model

def inference(model, input, VAL_AMP=False):
    def _compute(input_img):
        print(input_img.shape)
        input_img = torch.unsqueeze(input_img, 0)
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
    if isinstance(prediction, torch.Tensor):
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
