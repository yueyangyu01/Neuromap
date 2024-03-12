import torch
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
import json
import numpy as np


#
# res_net = SegResNet(
#     blocks_down=[1, 2, 2, 4],
#     blocks_up=[1, 1, 1],
#     init_filters=16,
#     in_channels=4,
#     out_channels=3,
#     dropout_prob=0.2,
# )
#
# res_net.load_state_dict(torch.load("/best_metric_model.pth"))
#
# res_net.eval()

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

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        input_dict = json.loads(request_body)
        img_data = input_dict['Body']
        img_bytes = img_data['data']
        shape = img_data['shape']
        dtype = img_data['dtype']
        data = np.frombuffer(img_bytes, dtype=dtype)
        data = data.reshape(shape)

        return data

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
        prediction = {
            'shape': pred_dims,
            'data': pred_bytes,
        }
    if accept == 'application/json':
        return json.dumps({'Body': prediction})