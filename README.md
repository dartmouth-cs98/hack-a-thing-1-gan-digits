### Deep Convolutional Generative Adversarial Network
#### For Digit Regocnition and Creation
##### Tom Young, 2018
##### From Felix Mohr's [Tutorial](https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a)
### What is it?
A generative adversarial network (DCGAN) is pairing between a descriminitive and generative neural network. As the descriminitive network is trained to recognize digits, the generative network tries to fool it by taking a latent vector of noise and performing convolutions on the vector in order to create an image that the descriminator might see as a number. If the descriminator fails the generator, it's loss function is used to adjust the weights on the generative network, allowing for "training" of the generative network.

![GAN](http://www.shashwatverma.com/assets/images/gans-cover.jpg "GAN")

![iteration 50](https://imgur.com/envIF9e.jpg "50")
![iteration 150](https://imgur.com/e0uvADk.jpg "150")
![iteration 500](https://imgur.com/P2xANqR.jpg "500")
![iteration 800](https://imgur.com/4i9uRV9.jpg "800")
![iteration 3000](https://imgur.com/VxguyMA.jpg "3000")
![iteration 12000](https://imgur.com/W975hvX.jpg "12000")

### What did I learn?

I used this project to learn two main things. The first was to learn how a GAN operates and why they work. This led me to learn about activation functions, loss functions, convolutions, gradient descent, etc. The second thing I learned while working on this project is how to use TensorFlow to actually implement neural networks. While I am still not 100% comfortable using the library, this was nonetheless a great step in the right direction. I also learned how to use Jupyter, and how to use IPython's "magic function expressions".

### What didn't work?

To my surprise, a lot more went right than went wrong. I was of course following a tutorial, but regardless I am glad to say that my issues were few and far between. The first problem I had was with getting TensorFlow installed in the first place, I was encoutering strange errors and eventually figured out that the version of python I was using (3.7) was not yet compatible with TesorFlow, so I installed another version of python (3.6.2) and got things working. The other problem I had was with MatPlotLib, which occured because I hadn't made the Pyplot interactive, causing the code to hang on the show() method of the plot. In order to fix this I used IPython's "magic functions" to make an inline plot that would be displayed with the output in Jupyter. This fixed the problem with the code hanging as soon as the first plot was ready to be displayed.

