# BeeGAN
The latest techniques in machine learning applied to bees!

## To Run:
### 1. Pre-Process the Audio Files:
Run the script located at `BeeGAN/Utils/PreProcessAudio.py`. An example invocation (from the project root directory) is 
provided below:
```cmd
python Utils/PreProcessAudio.py --root-data-dir "D:\data\Bees\beemon\raw\rpi4-2\2020-08-24\audio" --output-data-dir "D:\data\Bees\beemon\processed" --sample-rate 8000 -v
```
### 2. Run the PCA NN Sequential Script:
Run the script located at `BeeGAN/PCAGAN/PCANNSequential.py`. An example invocation (from the project root directory)
is provided below:
```cmd
python PCAGAN/PCANNSequential.py --data-file "D:\data\Bees\beemon\processed\transformed_data.npz" --num-components 20 -v
```