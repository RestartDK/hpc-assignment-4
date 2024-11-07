# Distributed Training with TensorFlow and Docker

This project is an in-class assignment demonstrating distributed training techniques with TensorFlow. It explores single-GPU, multi-GPU, and multi-node training setups. The project includes a Jupyter notebook with an initial question answered and a detailed report at the end summarizing the training results and insights.

## Project Structure

- **`notebook.ipynb`**: Contains an overview of distributed training, answers to the assignment's questions, and a final report comparing different training setups (single GPU, multi-GPU, and multi-node).
- **`main.py`**: The primary training script for distributed training. Configured to use TensorFlow’s `MultiWorkerMirroredStrategy` for multi-node training, with printed results at the end.
- **`mnist_setup.py`**: Defines the dataset loading, preprocessing, and model-building functions.
- **`Dockerfile`**: Docker configuration file for setting up the environment to simulate a multi-node setup.

## Requirements

- **Docker**: Used to create and manage containers for multi-node simulation.
- **Python 3.8+**: To run the Jupyter notebook locally if desired.

## Project Setup

To simulate the multi-node setup, Docker is used to create containers that emulate separate nodes. Each node runs as an independent container, and TensorFlow manages training distribution across these nodes.

### Step 1: Build the Docker Image

1. Clone the repository and navigate to the project directory.

   ```bash
   git clone https://github.com/RestartDK/hpc-assignment-4.git
   cd hpc-assignment-4/
   ```

2. Build the Docker image, which includes TensorFlow and other necessary dependencies.

   ```bash
   docker build -t tf_multi_worker .
   ```

### Step 2: Run Multi-Node Training with Docker

To simulate multiple nodes on a single machine, you’ll start two Docker containers, each representing a separate node.

1. Run each container with a unique `TF_CONFIG` environment variable, which instructs TensorFlow on the node configurations. Here’s how to run two containers as separate workers:

   - **Worker 0 (Chief)**:

     ```bash
     docker run -d --name worker0 \
         -e TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0}}' \
         -p 12345:12345 \
         tf_multi_worker_image
     ```

   - **Worker 1**:

     ```bash
     docker run -d --name worker1 \
         -e TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1}}' \
         -p 23456:23456 \
         tf_multi_worker_image
     ```

### Step 3: Monitor Training Progress and Results

Each container will print its training progress and results directly to the console. To view the logs and output, use:

```bash
docker logs -f worker0  # View logs and results for Worker 0
docker logs -f worker1  # View logs and results for Worker 1
```

After training completes, you’ll see each worker’s final results, including the total training time and test accuracy, directly in the output.

### Optional: Run the Jupyter Notebook

To explore the code, analysis, and summary report interactively, open `notebook.ipynb` in Jupyter Notebook:

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `notebook.ipynb` from the Jupyter interface.

---

## Key Points

- **Multi-Node Simulation**: The Docker setup allows for multi-node training emulation, letting you experiment with distributed training on a single machine.
- **Training Summary Report**: At the end of `notebook.ipynb`, a detailed report compares training performance across different setups.
- **Printed Results**: Results for training time and accuracy are printed directly in the logs of each container, accessible with `docker logs`.
- **Project Goals**: This project introduces distributed training using TensorFlow and Docker, providing insights into training time, accuracy, and implementation challenges across various distributed strategies.
