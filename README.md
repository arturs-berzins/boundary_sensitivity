# Neural Implicit Shape Editing using Boundary Sensitivity
**Arturs Berzins, Moritz Ibing, Leif Kobbelt**
### **[arXiv](https://arxiv.org/abs/2304.12951) | [SlidesLive](https://recorder-v3.slideslive.com/?share=79382&s=70b38e48-82c0-4104-9d01-18983330f331)**

## GUI Demo for Semantic Editing

The GUI is based on the [DualSDF demo](https://github.com/zekunhao1995/DualSDF).
The method works with different models, given a proper interface to the model to evaluate it.

Three pretrained models are available under `interfaces/DualSDF/pretrained/` based on the DualSDF generator: airplanes, cars, and chairs.


```bash
# Host the chair demo on the default port 1234
python demo.py interfaces/DualSDF --interface_args \
"./config/dualsdf_chairs_demo.yaml --pretrained ./pretrained/dualsdf_chairs_demo/epoch_2799.pth"
```
Then visit `localhost:1234`.
Type `python demo.py --help` to see and adjust the launch arguments.

https://user-images.githubusercontent.com/42326304/235196326-88f0550d-3b84-4758-9137-15215baa46fc.mp4

### TODO
- Add pre-trained IM-Net.
- Details on how to add your own model.
- Host the large pretrained model files not on GitHub

## Licence
MIT License
