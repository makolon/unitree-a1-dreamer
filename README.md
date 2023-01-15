# unitree-a1-dreamer
Unitree-A1-Dreamer

## Getting Started
We assume that you have access to a GPU with CUDA >= 11.0 support.
### Installation
1. Clone the repository
```
git clone https://github.com/makolon/unitree-a1-dreamer.git
```
2. Build docker image
```
cd ./docker
./build.sh
```
3. Run docker container
```
./run.sh
```

## Usage
```
cd unitree-a1-dreamer/dreamer \
python3 train.py --config defaults unitree --env_id UnitreeA1
```

## Visualize learning results
```
cd unitree-a1-dreamer/dreamer \
mlflow ui
```
Then, you can see the learning results by accessing http://127.0.0.1:5000
