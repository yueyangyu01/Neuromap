# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install MONAI
RUN pip install monai
RUN pip install nibabel
RUN pip install numpy
RUN pip install boto3


# Install any other dependencies your model requires
# RUN pip install <package-name>

# Copy the inference script into the container (optional)
# This is not strictly necessary if the inference.py is included in the model tarball
# COPY inference.py ./

# Set the entry point for the container
ENTRYPOINT ["python", "inference.py"]
