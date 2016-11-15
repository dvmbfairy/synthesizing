# from https://open_nsfw.gitlab.io/code.py

import settings
import caffe
import numpy as np

import numpy as np
import math, random
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow

caffe.set_mode_gpu()

def save_image(img, name):
  '''
  Normalize and save the image.
  '''
  img = img[:,::-1, :, :] # Convert from BGR to RGB
  normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))        
  scipy.misc.imsave(name, normalized_img)

def get_shape(data_shape):

  # Return (227, 227) from (1, 3, 227, 227) tensor
  if len(data_shape) == 4:
    return (data_shape[2], data_shape[3])
  else:
    raise Exception("Data shape invalid.")

np.random.seed(0)

generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)
shape = generator.blobs["feat"].data.shape
generator_output_shape = generator.blobs["deconv0"].data.shape

mean = np.float32([104.0, 117.0, 123.0])

net = caffe.Classifier("nets/placesCNN/places205CNN_deploy_updated.prototxt", 
                       "nets/placesCNN/places205CNN_iter_300000.caffemodel",
                       mean = mean,            # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

nsfw_net = caffe.Classifier("../open_nsfw/nsfw_model/deploy.prototxt", 
            		       "../open_nsfw/nsfw_model/resnet_50_1by2_nsfw.caffemodel",
                       mean = mean,            # ImageNet mean
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB


def grad_classifier(classifier, end_layer, imagein, z):

  net_dst = classifier.blobs[end_layer]

  # Do forward pass
  acts = classifier.forward(data=imagein, end=end_layer)

  # Do backwards pass
  net_dst.diff[:] = z
  g = classifier.backward(start=end_layer, diffs=['data'])['data'][0]

  # Cleanup
  net_dst.diff.fill(0.)
  return g, acts

def grad(classifier, end_layer, i, code):

  # Perform Forward Step
  generated = generator.forward(feat=code)
  image = crop(classifier, generated["deconv0"])

  # Set the inner product the gradient is taken w.r. to
  z = np.zeros_like(classifier.blobs[end_layer].data)
  z.flat[i] = 1

  # Do backwards step
  g, acts = grad_classifier(classifier, end_layer, image, z)
  generator.blobs['deconv0'].diff[...] = pad(classifier, g)
  gx = generator.backward(start='deconv0')

  # Cleanup
  generator.blobs['deconv0'].diff.fill(0.)
  return gx['feat'], image

def crop(classifier, image):
  data_shape  = classifier.blobs['data'].data.shape
  image_size  = get_shape(data_shape)
  output_size = get_shape(generator_output_shape)
  topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
  return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

def pad(classifier, image):
  data_shape  = classifier.blobs['data'].data.shape
  image_size  = get_shape(data_shape)
  output_size = get_shape(generator_output_shape)
  topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
  o = np.zeros(generator_output_shape)
  o[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = image
  return o

"""
Generate Random Picture
"""
#for category in [0, 111, 184, 199, 171, 140, 78] :

w2 = 1000

for k in range(200,300):

  for category in [0, 7, 8, 11, 34, 36, 59, 68, 80, 96, 97, 106, 40, 126 ,128, 184, 157, 174, 179, 192]:
    np.random.seed(k)
    code = np.random.normal(0, 1, shape)

    total_iters = 300

    alpha = 1
    # Load the activation range
    upper_bound = lower_bound = None

    # Set up clipping bounds
    upper_bound = np.loadtxt("act_range/3x/fc6.txt", delimiter=' ', usecols=np.arange(0, 4096), unpack=True)
    upper_bound = upper_bound.reshape(4096)

    # Lower bound of 0 due to ReLU
    lower_bound = np.zeros(4096)

    for i in range(0,total_iters):
      step_size = (alpha + (1e-10 - alpha) * i) / total_iters
      gp, image = grad(net, 'fc8', category, code)
      gp = gp.copy()
      gn, image = grad(nsfw_net, 'prob', 1, code)
      
      # To generate NSFW Samples
      # Lowering the stepsize to 0.5 AND using g = 0.1*gp + 1500*gn is also good
      if category in set([197, 157]):
        w2 = 2100
      elif category in set([34, 40, 126]):
        w2 = 1400
      else: 
        w2 = 1200

      g = 0.1*gp + w2*gn
      
      # To generate Regular Samples
      # g = 1*gp + 0*gn

      print norm(gp), norm(gn)

      if norm(g) <= 1e-8:
        break
      code = code - step_size*g/np.abs(g).mean()
      code = np.maximum(code, lower_bound) 

      # 1*upper bound produces realistic looking images
      # No upper bound produces dramatic high saturation pics
      # 1.5* Upper bound is a decent choice
      code = np.minimum(code, 1.5*upper_bound) 

    save_image(image, "final_places/w_" + str(w2) + "_" + str(category) + "_" + str(k) + ".jpg")

