# Use MONAI as a parent image
FROM projectmonai/monai:latest

# Install deps
RUN pip install nibabel
RUN pip install numpy
RUN pip install boto3
RUN pip install matplotlib

# Install any other dependencies your model requires
# RUN pip install <package-name>

# Copy the inference script into the container (optional)
# This is not strictly necessary if the inference.py is included in the model tarball
# COPY inference.py ./

# Set the entry point for the container
ENTRYPOINT ["python", "inference.py"]
