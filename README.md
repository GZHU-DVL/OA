Our code uses PyTorch (pytorch >= 0.4.1, torchvision >= 0.2.1) with CUDA10.1 and Python 3.7. The script run_simba.py contains code to run SimBA  with various options.

To run Object-attentional-attack:
```
python run_Object_Attentional.py --data_root <imagenet_root> --num_iters 10000 --pixel_attack  --freq_dims 224
```