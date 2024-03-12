import nibabel as nib
import boto3
import numpy as np
import json

# TODO: fix this function
def nifti_to_json(nifti_file):
    img = nib.load(nifti_file)
    img_data_type = img.get_data_dtype()
    # data to np array
    image_data = img.get_fdata()
    # data to bytes
    serialized_data = image_data.tobytes()

    json_data = json.dumps({
        'shape': image_data.shape,
        'dtype': str(img_data_type),
        'data': serialized_data,
    })

    return json_data


def bytes_to_nifti(img_bytes, shape: tuple, dtype: str = 'float32'):
    data = np.frombuffer(img_bytes, dtype=eval(dtype))
    data = data.reshape(shape)
    img = nib.Nifti1Image(data, np.eye(data.shape[0]))
    return img

def invoke_sagemaker_endpoint(payload, endpoint_name=''):
    '''
    Invokes a SageMaker endpoint with the given payload
    Payload: json body
    EndpointName: str
    '''
    client = boto3.client('sagemaker-runtime')
    content_type = 'application/octet-stream'

    try:
        # Invoke the SageMaker endpoint
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload
        )

        # Read the response payload
        response_body = response['Body'].read()
        return response_body

    except Exception as e:
        # Handle errors, such as EndpointNotFoundException or ValidationError
        print(f"Error invoking SageMaker endpoint: {e}")

def process_endpoint_response(response_body):
    # Convert the response from bytes to a JSON string
    reponse_dict = json.load(response_body)
    reponse_data = reponse_dict['data']
    reponse_shape = reponse_dict['shape']
    reponse_dtype = reponse_dict['dtype']

    # Convert the response data to a NIfTI image
    nifti_img = bytes_to_nifti(reponse_data, reponse_shape, reponse_dtype)

    return nifti_img


