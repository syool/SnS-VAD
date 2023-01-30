## Environments
* Python 3.8
* PyTorch 1.10.0

Import Docker envrionment using Dockerfile

## Datasets
* UCSD Pedestrian 2
* CUHK Avenue
* ShanghaiTech Campus

## Training & Inference
```bash
# For training
python3 main.py --dataset ped2 --cuda 0 --train
```
```bash
# For inference
python3 main.py --dataset ped2 --cuda 0 --inference
```