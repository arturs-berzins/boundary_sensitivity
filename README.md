## Neural Implicit Shape Editing using Boundary Sensitivity
### **[arXiv](https://arxiv.org/abs/2304.12951) | [Presentation & Poster](https://iclr.cc/virtual/2023/poster/11096) | [OpenReview](https://openreview.net/forum?id=CMPIBjmhpo)**

https://user-images.githubusercontent.com/42326304/235196326-88f0550d-3b84-4758-9137-15215baa46fc.mp4

This is a demo of semantic editing as described in the ICLR 2023 paper _Neural Implicit Shape Editing using Boundary Sensitivity_.
The method allows to perform semantically meaningful editing of shapes using an already trained shape-generative model by simply specifying a few design handles!


The GUI is based on the [DualSDF demo](https://github.com/zekunhao1995/DualSDF). Three pretrained models are available under `interfaces/DualSDF/pretrained/` based on the DualSDF generator trained on airplanes, cars, and chairs. To run:


```bash
# Host the chair demo on the default port 1234
python demo.py interfaces/DualSDF --interface_args \
"./config/dualsdf_chairs_demo.yaml --pretrained ./pretrained/dualsdf_chairs_demo/epoch_2799.pth"
```
Then visit `localhost:1234`.
Type `python demo.py --help` to see and adjust the launch arguments.

The editing computation itself is extremely cheap. The bulk of compute time spent doing marching cubes to extract the surface mesh from the implicit function.

## Requirements
The only requirements are `numpy`, `torch`, `scikit-image` for marching cubes, `pyyaml` for reading `yaml` files for DualSDF. The tested versions are given in `requirements.txt`, but these should be quite flexible.

## Other models
The method works with any shape-generative model. To implement this in the code, you must provide a proper interface to the model: see `interfaces/interface_base.py` and `interfaces/DualSDF/interface.py` for more information and an example.


## Cite
```
@inproceedings{berzins23bs,
  title={Neural Implicit Shape Editing using Boundary Sensitivity},
  author={Berzins, Arturs and Ibing, Moritz and Kobbelt, Leif},
  booktitle={The Eleventh International Conference on Learning Representations},
  publisher={OpenReview.net},
  year={2023}
}
```