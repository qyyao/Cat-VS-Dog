## CNN model to recognize images of cats and dogs
This is a CNN project predict whether an image holds a dog or a cat

### The dataset:
[Dogs vs. Cats Kaggle Competition](https://www.kaggle.com/c/dogs-vs-cats/data)
* Training dataset:
    * 1000 dogs
    * 1000 cats
* Validation dataset
    * 500 dogs
    * 500 cats
* Testing dataset:
    * 1000 combined

### Project considerations:
* Binary classification problem: cat or dog
* Coloured pictures means we must have three channels, one for each of RGB
* Picture dimensions are not standardized, must convert 
* Data selected is a subset of full kaggle dataset, and number of epochs is lowered due to personal computing power. Ideally, we would use higher epoch count for higher accuracy.

### Results and Optimization:
When our model was ran without optimization, we achieved the following:
![Alt text](output_graphs/default_model.png?raw=true "Non-optimized Output")

While the accuracy continues to increase, validation accuracy begins to plateau. This suggests our data is over-fitted.

1) Image augmentation:

    To resolve the problem of over-fitting, I applied augmentation to the ImageDataGenerator function:
    * rotation
    * width shift
    * height shift
    * shearing
    * zooming

    Allowing us to achieve the following:        
![Alt text](output_graphs/augmented_model.png?raw=true "Augmented Output")

    It is shown that validation accuracy now climbs together with accuracy. However, by the cut off at the 10th epoch, the accuracy is still not very high. We can hypothesis the accuracy will continue to grow if epoch count is increased.

2) Increasing Epoch Count:

    Both our above models only use 10 epoch due to my own personal computing power at the moment, but can be increased in order to allow our accuracy to continue growing
    
3) Transfer learning (VGG16):

    VGG16 is a convolutional model holding different animal species. As we use dogs and cats in our own model, we may utilize this pre-built model as they share data and features.
    Since training this model is very time-consuming, this output will be updated at a later time.
    