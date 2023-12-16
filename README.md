# Anomaly detection with fast calorimeter simulators
## by Claudius Krause, Benjamin Nachman, Ian Pang, David Shih and Yunhao Zhu
This repository contains the source code used to produce results of

_"Anomaly detection with fast calorimeter simulators"_ by Claudius Krause, Benjamin Nachman, Ian Pang, David Shih and Yunhao Zhu, [arxiv: 2312.xxxx]

### Detector Layout and Training Data
We consider a new sampling calorimeter version of the toy detector used in the original [CaloGAN](https://arxiv.org/abs/1712.10321). The original dataset included energy contributions from both active and inactive calorimeter layers, whereas our new dataset only includes energy contributions from the active layers as would be available in practice. The sampling fraction of our calorimeter setup is $\sim20\%$. Like the original toy detector, we have a three-layer, liquid Argon (LAr) electromagnetic calorimeter (ECal) cube with 480mm side length. The three layers have a voxel resolution of $`3\times 96`$, $`12\times 12`$, and $`12\times 6`$, respectively. 

The new dataset can be found at https://zenodo.org/records/10393540.
