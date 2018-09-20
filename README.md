### Deep Convolutional Generative Adversarial Network
#### For Digit Regocnition and Creation

### What is it?
A deep convolutional generative adversarial network (DCGAN) is pairing between a descriminitive and generative neural network. As the descriminitive network is trained to recognize digits, the generative network tries to fool it by taking a latent vector of noise and performing convolutions on the vector in order to create an image that the descriminator might see as a number. If the descriminator fails the generator, it's loss function is used to adjust the weights on the generative network, allowing for "training" of the generative network.

![GAN](http://www.shashwatverma.com/assets/images/gans-cover.jpg "GAN")

![iteration 50](https://imgur.com/envIF9e.jpg "50")
![iteration 150](https://imgur.com/e0uvADk.jpg "150")
![iteration 500](https://imgur.com/P2xANqR.jpg "500")
![iteration 800](https://imgur.com/4i9uRV9.jpg "800")
![iteration 3000](https://imgur.com/VxguyMA.jpg "3000")
![iteration 12000](https://imgur.com/W975hvX.jpg "12000")


