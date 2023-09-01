# CL-SegFormer
Retraining a SegFormer with a branch that generates query and key embeddings from intermediate hierarchical feature map logits. This is done so that a contrastive learning task can be performed alongside the segmentation model.  

![Alt Text](/Unknown.png)


* Please note that experimentation is not reproducible by simply running "train.py", but the code is structured in such a way that you have access to everything you need to carry out experimentation.

* Be sure to adjust all root directories in the scripts. 

* CityScapes dataset follows a dataset structure as follows:

~~~text
CityScapes/
├── annotation
│   └── city_gt_fine
│       ├── train
│       └── val
└── images
    └── city_gt_fine
        ├── train
        └── val
~~~

The authors of this work would like to give significant thanks to the authors of 
* SegFormer, (Xie, E., Wang, W., Yu, Z., Anandkumar, A., Álvarez, J.M. and Luo, P., 2021. Segformer:
Simple and eﬃcient design for semantic segmentation with transformers. Corr [Online],
abs/2105.15203. 2105.15203, Available from: https://arxiv.org/abs/2105.15203.)
* PEBAL, (Tian, Y., Liu, Y., Pang, G., Liu, F., Chen, Y. and Carneiro, G., 2021. Pixel-wise
energy-biased abstention learning for anomaly segmentation on complex urban driving
scenes. Corr [Online], abs/2111.12264. 2111.12264, Available from: https://arxiv.
org/abs/2111.12264)
* Wang, H., Lu, Y. and Chen, X., 2023. Contrastive vision transformer for self-supervised
out-of-distribution detection [Online]. Available from: https://openreview.net/
forum?id=UAmH4nDH4l.
  
