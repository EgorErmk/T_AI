import matplotlib.pyplot as plt

import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
image = keras_ocr.tools.read('https://opengraph.githubassets.com/4e1af165bf1e4906e750c0118cad2cb0767bb506e4539977d53bdd7810bd9037/tensorflow/io/issues/1087')

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize([image])

# Plot the predictions
fig, axs = plt.subplots(nrows=len([image]), figsize=(20, 20))
for ax, img, predictions in zip([axs], [image], prediction_groups):
    keras_ocr.tools.drawAnnotations(image=img, predictions=predictions, ax=ax)
fig.savefig('image.png')