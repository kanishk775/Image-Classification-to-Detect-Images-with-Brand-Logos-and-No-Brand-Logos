# Image-Classification-to-Detect-Images-with-Brand-Logos-and-No-Brand-Logos
We will use Convolutional Neural Network to classify the pictures. The fundamental difference between a densely connected layer and a convolution layer is this:
Dense layers learn global patterns in their input feature space, whereas convolution layers learn local patterns—in the case of images, patterns found in small
2D windows of the inputs.


Initially, I started training my own convolutional model with the image dataset that I had without using any off the shelf trained network. We get a test 
accuracy of 48.4%. Because we have relatively few training samples, overfitting will be our number one concern. The training accuracy stabilises at nearly 72%
and then increases, whereas the validation accuracy stabilises at 64%. The validation loss reaches its minimum after only three epochs and then increases,
whereas the training loss keeps decreasing linearly as training proceeds.

To Fine Tune the data, I use data augmentation.The model will never see the same input twice. But the inputs it sees are still heavily intercorrelated because
they come from a small number of original images—we can’t produce new information; we can only remix existing information. As such, this may not be enough to
completely get rid of overfitting.

By further tuning the model’s configuration (such as the number of filters per convolution layer, or the number of layers in the model), we might be able to
get an even better accuracy, likely up to 90%. But it would prove difficult to go any higher just by training our own convnet from scratch, because we have
so little data to work with. As a next step to improve our accuracy on this problem, we’ll have to use a pretrained model.

Pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is
large enough and general enough, the spatial hierarchy of features learned by the pretrained model can effectively act as a generic model of the visual world,
and hence, its features can prove useful for many different computer vision problems, even though these new problems may involve completely different classes
than those of the original task.

We get a test accuracy of 73.4%. The training accuracy stabilises at nearly 95% and then increases, whereas the validation accuracy oscillates at 82%. The
validation loss reaches its minimum after only two epochs and then increases, whereas the training loss keeps decreasing linearly as training proceeds.



What went well? What didn't go well?

If we train a new model using this data-augmentation configuration, the model will never see the same input twice. But the inputs it sees are still heavily
intercorrelated because they come from a small number of original images—we can’t produce new information; we can only remix existing information. As such,
this may not be enough to completely get rid of overfitting.A number of techniques that can help mitigate overfitting are dropout and weight decay
(L2 regularization).
Convnets are the best type of machine learning models for computer vision tasks. It’s possible to train one from scratch even on a very small dataset,
with decent results. Convnets work by learning a hierarchy of modular patterns and concepts to represent the visual world.
On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.
It’s easy to reuse an existing convnet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets. 
As a complement to feature extraction, we used fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model.
This pushes performance a bit further.

What are your system's limitations?

Image Dataset was too low. Having a larger dataset would help in distributing more images to train, test, validation split.
Convolutional layers could be increased to enhance the learning and increase accuracy.
Google Colab used to take a lot of time to run the epochs. Having a premium account for Google Colab could increase computng speed.
Instead of VGG16 which has become old, Efficient Net or Mobile Net could be used.
