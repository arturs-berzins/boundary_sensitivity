
Pretrained model from [Learning Implicit Fields for Generative Shape Modeling](https://github.com/czq142857/IM-NET-pytorch).
Removed many unnecessary parts, like Single-View-Reconstruction, training, testing, data.

The pretrained models have originally been acquired as follows:
- Train an AE on voxels in a binary classification task. Encoder is CNN, decoder is IM.
- Encode the training set shapes into latent codes.
- Use these latent codes to train a GAN able to generate new latent codes.
- Use the generator G from the GAN to generate novel latent codes.

Unfortunately, the GAN code or trained weights are not provided, as also indicated in [this issue](https://github.com/czq142857/IM-NET-pytorch/issues/6).
The best option seems to be to implement a `pytorch` model and take the `tensorflow` weights.