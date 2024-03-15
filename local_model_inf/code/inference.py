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
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

import numpy as np
import boto3
import json
import io
import os
import tempfile
from skimage.measure import marching_cubes
import trimesh


# TESTING: DONE AND WORKS
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

        # Temporarily download image from S3
        if 'nii.gz' in object_key :

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, object_key)
                # print(local_path)
                
                # Create the necessary directories for the local_path
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                s3.download_file(bucket_name, object_key, local_path)

                # Load the image
                img = nib.load(local_path)
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

# TESTING: DONE
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

# TODO: Fix
def send_to_s3(data, bucket, key):
    data = torch.tensor(data)
    s3 = boto3.client('s3')
    s3.put_object(Body=data, Bucket=bucket, Key=key)


def inf_to_nifti(pred, patient_id):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    prediction = nib.load(prediction)
    save_directory = os.path.join('/local_data/predictions', str(patient_id))
    nib.save(prediction, save_directory)

def pytorch_to_stl(prediction):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
        inf_0 = prediction[0, 0, :, :, :]
        inf_1 = prediction[0, 1, :, :, :]
        inf_2 = prediction[0, 2, :, :, :]

        verts_0, faces_0, normals_0, values_0 = marching_cubes(inf_0, level=0.5)
        verts_1, faces_1, normals_1, values_1 = marching_cubes(inf_1, level=0.5)
        verts_2, faces_2, normals_2, values_2 = marching_cubes(inf_2, level=0.5)

        mesh_0 = trimesh.Trimesh(vertices=verts_0, faces=faces_0)
        mesh_1 = trimesh.Trimesh(vertices=verts_1, faces=faces_1)
        mesh_2 = trimesh.Trimesh(vertices=verts_2, faces=faces_2)

        return mesh_0, mesh_1, mesh_2
    raise TypeError("Input must be a torch.Tensor")
    
def export_stl(mesh, mesh_num, patient_id):
    # If patient directory doesn't exist, create it
    if not os.path.exists(os.path.join('local_data/predictions', str(patient_id))):
        os.makedirs(os.path.join('local_data/predictions', str(patient_id)))
    save_directory = os.path.join('local_data/predictions', str(patient_id), str(mesh_num))
    mesh.export(save_directory+f'{patient_id}.stl')


def local_output(prediction, accept='application/json'):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if accept == 'application/json':
        # Turn into blob
        prediction_blob = 0
        prediction_dtype = str(prediction.dtype)
        # eval(prediction_dtype)
        prediction_shape = prediction.shape

        prediction = {
            'shape': prediction_shape,
            'data': prediction_blob,
            'dtype': prediction_dtype,
        }

        return json.dumps({'Body': prediction})


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


# TESTING

# Load the model -- WORKS
# Go back one directory to the root of the project
os.chdir('..')
model = model_fn('./')
print(model.__class__)

# Load the image paths -- WORKS
test_paths = list_image_paths('bobmarley')
print(test_paths)

# Create images 
imgs, og_headers = transform_data(test_paths)
print(imgs.shape)

# Inference
pred = inference(model, imgs)
print(pred.shape)
# Only need outputs 0 and 1 (WT and ET)

# Send to S3
# send_to_s3(pred, 'neuromapuserimages', 'danazarezankova/prediction.nii.gz')
# print("sent to S3")

# Output
# Goal: convert pytorch to stl
mesh_0, mesh_1, mesh_2 = pytorch_to_stl(pred)
# O: Whole tumor 1: Tumor core 2: Enhanced tumor
export_stl(mesh_0, 0, 'bobmarley')
export_stl(mesh_1, 1, 'bobmarley')
export_stl(mesh_2, 2, 'bobmarley')
print("exported to stl")