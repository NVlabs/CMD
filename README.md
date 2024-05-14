<h1 align="center"> Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition</h1>
<div align="center">
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://weilinie.github.io/" target="_blank">Weili&nbsp;Nie</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://ai.stanford.edu/~dahuang/" target="_blank">De-An&nbsp;Huang</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/site/boyilics/home" target="_blank">Boyi&nbsp;Li</a><sup>2,3</sup>
  <br>
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="http://tensorlab.cms.caltech.edu/users/anima/" target="_blank">Anima&nbsp;Anandkumar</a><sup>4</sup><br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>NVIDIA Corporation &emsp; <sup>3</sup>UC Berkerley &emsp; <sup>4</sup>Caltech &emsp; <br>
</div>
<h3 align="center">[<a href="https://sihyun.me/CMD">project page</a>] [<a href="https://openreview.net/forum?id=dQVtTdsvZH">openreview</a>]</h3>


### 1. Environment setup
```bash
conda create -n cmd python=3.8 -y
conda activate cmd
pip install -r requirements.txt
```

### 2. Dataset 

#### Dataset download
Currently, we provide experiments for [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php). You can place the data that you want and can specifiy it via `--data-path` arguments in training scripts.

#### UCF-101
```
UCF-101
|-- class1
    |-- video1.avi
    |-- video2.avi
    |-- ...
|-- class2
    |-- video1.avi
    |-- video2.avi
    |-- ...
    |-- ...
```


### 3. Training

#### Autoencoder

```bash
 torchrun --nnodes=[NUM_NODES] --nproc_per_node=[NUM_GPU] train_ae.py \
    --dataset-name UCF101 \ 
    --data-path /data/UCF-101 \
    --global-batch-size [BATCH_SIZE] \ 
    --results-dir [LOG_DIRECTORY] 
    --mode pixel \
    --ckpt-every 20000
```

#### Motion Diffusion Model

```bash
python train_motion_diffusion.py \ 
  --nnodes=[NUM_NODES] \ 
  --nproc_per_node=[NUM_GPUS] 
    --dataset-name UCF101 \ 
    --data-path /data/UCF-101 \
    --global-batch-size [BATCH_SIZE] \   
    --results-dir [LOG_DIRECTORY]
    --mode pixel \
    --ckpt-every 20000
```

#### Content Diffusion Model

```bash
python train_content_diffusion.py \ 
  --nnodes=[NUM_NODES] \ 
  --nproc_per_node=[NUM_GPUS] 
    --dataset-name UCF101 \ 
    --data-path /data/UCF-101 \
    --global-batch-size [BATCH_SIZE] \   
    --results-dir [LOG_DIRECTORY]
    --mode pixel \
    --ckpt-every 20000 \
    --motion-model-config [MOTION_MODEL_CONFIG]
```

Then these scripts will automatically create the folder in `[LOG_DIRECTORY]` to save logs and checkpoints.

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.


## Citation

Please consider citing CMD if this repository is useful for your work. 

```bibtex
@inproceedings{yu2024cmd,
  title={Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition},
  author={Sihyun Yu and Weili Nie and De-An Huang and Boyi Li and Jinwoo Shin and Anima Anandkumar},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```


## Licenses

Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

## Acknowledgement

This code is mainly built upon [PVDM](https://github.com/sihyun-yu/PVDM), [DiT](https://github.com/facebookresearch/DiT), and [glide-text2im](https://github.com/openai/glide-text2im) repositories.\
We also used the code from following repositories: [StyleGAN-V](https://github.com/universome/stylegan-v) and [TATS](https://github.com/songweige/TATS).
