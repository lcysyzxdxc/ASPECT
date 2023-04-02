# RT-HQoE: 

A PYTHON implementation of real-time QoE model for adaptive video streaming (RT-HQoE) in [IEEE ICME2023] [A REAL-TIME BLIND QUALITY-OF-EXPERIENCE ASSESSMENT METRIC FOR HTTP ADAPTIVE STREAMING](https://arxiv.org/pdf/2303.09818).

## 1. Introduction

HTTP Adaptive Streaming (HAS) is the dominant delivery method for Video on Demand (VoD) services. An effective Quality of Experience (QoE) assessment metric can provide crucial feedback to its Adaptive BitRate (ABR) algorithm. 

However, predicting such real-time QoE on the client side is challenging. The QoE prediction requires high consistency with the Human Visual System (HVS), low latency, and blind assessment, which are difficult to realize together.

Therefore, we design an effective QoE metric that integrates resolution and rebuffering time as the Quality of Service (QoS), as well as spatio-temporal output from a deep neural network and specific switching events as content information, regressed with a Support Vector Regression (SVR) model. 

## 2. Demos

#### Requirement

A. An mp4 file of a video streaming with 5 HAS chunks.

B. A csv file including those essential information of video:
```
rebuffering_duration	chunk_duration	framerate
```
   And at least one of those QoS features:
```
video_bitrate	qp	width	height
```

C. Python packages:
```
python 3.8.8
skvideo 1.1.11
numpy 1.22.4
pandas 1.5.3
torch 1.8.1
torchvision 0.9.1
skimage 0.19.3
scipy 1.6.2
argparse 1.1
```

#### Feature Extraction

```
python demo.py -q [QoS feature] -s [global sampling intensity]
```
The QoS feature can be video height, width, QP, bitrate, etc. You can also use your own QoS feature, but remember to update its csv flie index.

A higher global sampling intensity can provide better result with more complexity. Generally 20~30 can realize real-time.

#### Quality Prediction

Please download [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) for regression. 

Note: Our original model needs manual operation, based on both MATLAB and PYTHON. So in this executable PYTHON code, the correlation factor is same as the paper result, but the computational time is slightly different. The automatic quality prediction still need MATLAB LIBSVM above, we will update its PYTHON version later.

## 3 Citation

If you find this work useful, please cite our paper as:

```
@misc{li2023realtime,
      title={A real-time blind quality-of-experience assessment metric for HTTP adaptive streaming}, 
      author={Chunyi Li and May Lim and Abdelhak Bentaleb and Roger Zimmermann},
      year={2023},
      eprint={2303.09818},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Contact
Chunyi Li, ```lcysyzxdxc@sjtu.edu.cn```
