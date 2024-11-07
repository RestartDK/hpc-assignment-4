# Dockerfile
FROM armswdev/tensorflow-arm-neoverse:latest

# Set the environment variable to use legacy Keras
ENV TF_USE_LEGACY_KERAS=1

# Set the working directory
WORKDIR /app

# Copy the Python files into the container
COPY main.py mnist_setup.py /app/

# Install any additional dependencies if required (in this case, none)

# Set the entrypoint to run the main script
ENTRYPOINT ["python", "main.py"]
