import wandb
import mzutils

mzutils.mkdir_p('./wandb')
with wandb.init(project = '1', sync_tensorboard = True, dir='./wandb') as run:
    print(1)