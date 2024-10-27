# Neural Implicit Shape Editing using Boundary Sensitivity
**Arturs Berzins, Moritz Ibing, Leif Kobbelt**
### **[arXiv](https://arxiv.org/abs/2304.12951) | [Presentation & Poster](https://iclr.cc/virtual/2023/poster/11096) | [OpenReview](https://openreview.net/forum?id=CMPIBjmhpo)**

## GUI Demo for Semantic Editing

The GUI is based on the [DualSDF demo](https://github.com/zekunhao1995/DualSDF).

Three pretrained models are available under `interfaces/DualSDF/pretrained/` based on the DualSDF generator trained on airplanes, cars, and chairs. To run:


```bash
# Host the chair demo on the default port 1234
python demo.py interfaces/DualSDF --interface_args \
"./config/dualsdf_chairs_demo.yaml --pretrained ./pretrained/dualsdf_chairs_demo/epoch_2799.pth"
```
Then visit `localhost:1234`.
Type `python demo.py --help` to see and adjust the launch arguments.

https://user-images.githubusercontent.com/42326304/235196326-88f0550d-3b84-4758-9137-15215baa46fc.mp4


The method and code works with different models, given a proper interface to the model to evaluate it. See `interfaces/interface_base.py` and `interfaces/DualSDF/interface.py` for more information and an example.

The only requirements are `numpy`, `torch`, `scikit-image` for marching cubes, `pyyaml` for reading `.yaml` files for DualSDF. The tested versions are given in `requirements.txt`, but these should be quite flexible.

The bulk of compute time spent doing marching cubes to extract the surface mesh from the implicit function.

## Licence
MIT License
