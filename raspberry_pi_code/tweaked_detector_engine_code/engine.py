# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detection Engine used for detection tasks."""

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import image_processing
import numpy as np
from PIL import Image

print('yo')

ALLOWED_LABELS = [1,2,3,5,7]

class DetectionCandidate(object):
  """Data structure represents one detection candidate."""
  __slots__ = ['label_id', 'score', 'bounding_box']

  def __init__(self, label_id, score, x1, y1, x2, y2):
    #: int, label id.
    self.label_id = label_id
    #: float, score of the candidate.
    self.score = score
    #: numpy.array, describing the bouding box with format
    #:  [[x1, y1], [x2, y2]]. Where (x1,y1) is the top-left corner and (x2,y2)
    #:  is the bottom-right corner of the bounding box. The type of element can
    #:  be either float or integer, depending on relative_coord passed by user.
    self.bounding_box = np.array([[x1, y1], [x2, y2]])


class DetectionEngine(BasicEngine):
  """Engine used for detection tasks."""

  def __init__(self, model_path, device_path=None):
    """Creates a DetectionEngine with given model.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.
      device_path: String, if specified, bind engine with Edge TPU at device_path.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    if device_path:
      super().__init__(model_path, device_path)
    else:
      super().__init__(model_path)
    output_tensors_sizes = self.get_all_output_tensors_sizes()
    if output_tensors_sizes.size != 4:
      raise ValueError(
          ('Dectection model should have 4 output tensors!'
           'This model has {}.'.format(output_tensors_sizes.size)))
    self._tensor_start_index = [0]
    offset = 0
    for i in range(3):
      offset = offset + output_tensors_sizes[i]
      self._tensor_start_index.append(offset)

  def DetectWithImage(self, img, threshold=0.1, top_k=3,
                      keep_aspect_ratio=False, relative_coord=True,
                      resample=Image.NEAREST):
    """Detects object with given PIL image object.

    This interface assumes the loaded model is trained for object detection.

    Args:
      img: PIL image object.
      threshold: float, threshold to filter results. Default value = 0.1.
      top_k: keep top k candidates if there are many candidates with score
        exceeds given threshold. By default we keep top 3.
      keep_aspect_ratio: bool, whether to keep aspect ratio when down-sampling
        the input image. By default it's false.
      relative_coord: whether to converts coordinates to relative value. By
        default is true, all coordinates will be coverted to a float number
        in range [0, 1] according to width/height. Otherwise coordinates will
        be integers representing number of pixels.
      resample: An optional resampling filter on image resizing. By default it
        is PIL.Image.NEAREST. Complex filter such as PIL.Image.BICUBIC will
        bring extra latency, and slightly better accuracy.

    Returns:
      List of DetectionCandidate.

    Raises:
      RuntimeError: when model's input tensor format is invalid.
    """
    input_tensor_shape = self.get_input_tensor_shape()
    if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
        input_tensor_shape[0] != 1):
      raise RuntimeError(
          'Invalid input tensor shape! Expected: [1, height, width, 3]')
    _, height, width, _ = input_tensor_shape

    if keep_aspect_ratio:
      resized_img, ratio = image_processing.ResamplingWithOriginalRatio(
          img, (width, height), resample)
    else:
      resized_img = img.resize((width, height), resample)

    input_tensor = np.asarray(resized_img).flatten()
    candidates = self.DetectWithInputTensor(input_tensor, threshold, top_k)
    for c in candidates:
      if keep_aspect_ratio:
        c.bounding_box = c.bounding_box / ratio
        c.bounding_box[0] = np.maximum([0.0, 0.0], c.bounding_box[0])
        c.bounding_box[1] = np.minimum([1.0, 1.0], c.bounding_box[1])
      if relative_coord is False:
        c.bounding_box = c.bounding_box * [img.size]
    return candidates

  def DetectWithInputTensor(self, input_tensor, threshold=0.1, top_k=3):
    """Detects objects with raw input.

    This interface allows user to process image outside the engine for
    efficiency concern.

    Args:
      input_tensor: numpy.array represents the input tensor.
      threshold: float, threshold to filter results. Default value = 0.1.
      top_k: keep top k candidates if there are many candidates with score
        exceeds given threshold. By default we keep top 3.

    Returns:
      List of DetectionCandidate.

    Raises:
      ValueError: when input param is invalid.
    """
    if top_k <= 0:
      raise ValueError('top_k must be positive!')
    _, raw_result = self.RunInference(input_tensor)
    result = []
    num_candidates = raw_result[self._tensor_start_index[3]]
    for i in range(int(round(num_candidates))):
      score = raw_result[self._tensor_start_index[2] + i]
      if score > threshold:
        label_id = int(round(raw_result[self._tensor_start_index[1] + i]))
        if label_id in ALLOWED_LABELS:
            print('prediicted', label_id)
            y1 = max(0.0, raw_result[self._tensor_start_index[0] + 4 * i])
            x1 = max(0.0, raw_result[self._tensor_start_index[0] + 4 * i + 1])
            y2 = min(1.0, raw_result[self._tensor_start_index[0] + 4 * i + 2])
            x2 = min(1.0, raw_result[self._tensor_start_index[0] + 4 * i + 3])
            result.append(DetectionCandidate(label_id, score, x1, y1, x2, y2))
    result.sort(key=lambda x: -x.score)
    return result[:top_k]



import tensorflow as tf

class TfLiteDetectionEngine():
  """Engine used for detection tasks with tflite model."""

  def __init__(self, model_path):
    """Creates a DetectionEngine with given model.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.
      device_path: String, if specified, bind engine with Edge TPU at device_path.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    
    self.interpreter = tf.lite.Interpreter(model_path=model_path)
    self.interpreter.allocate_tensors()
    
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    

    output_tensors_sizes = self.output_details
    if len(output_tensors_sizes) != 4:
      raise ValueError(
          ('Dectection model should have 4 output tensors!'
           'This model has {}.'.format(len(output_tensors_sizes))))

  def DetectWithImage(self, img, threshold=0.1, top_k=3,
                      keep_aspect_ratio=False, relative_coord=True,
                      resample=Image.NEAREST):
    """Detects object with given PIL image object.

    This interface assumes the loaded model is trained for object detection.

    Args:
      img: PIL image object.
      threshold: float, threshold to filter results. Default value = 0.1.
      top_k: keep top k candidates if there are many candidates with score
        exceeds given threshold. By default we keep top 3.
      keep_aspect_ratio: bool, whether to keep aspect ratio when down-sampling
        the input image. By default it's false.
      relative_coord: whether to converts coordinates to relative value. By
        default is true, all coordinates will be coverted to a float number
        in range [0, 1] according to width/height. Otherwise coordinates will
        be integers representing number of pixels.
      resample: An optional resampling filter on image resizing. By default it
        is PIL.Image.NEAREST. Complex filter such as PIL.Image.BICUBIC will
        bring extra latency, and slightly better accuracy.

    Returns:
      List of DetectionCandidate.

    Raises:
      RuntimeError: when model's input tensor format is invalid.
    """
    raise NotImplementedError

    input_tensor_shape = self.get_input_tensor_shape()
    if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
        input_tensor_shape[0] != 1):
      raise RuntimeError(
          'Invalid input tensor shape! Expected: [1, height, width, 3]')
    _, height, width, _ = input_tensor_shape

    if keep_aspect_ratio:
      resized_img, ratio = image_processing.ResamplingWithOriginalRatio(
          img, (width, height), resample)
    else:
      resized_img = img.resize((width, height), resample)

    input_tensor = np.asarray(resized_img).flatten()
    candidates = self.DetectWithInputTensor(input_tensor, threshold, top_k)
    for c in candidates:
      if keep_aspect_ratio:
        c.bounding_box = c.bounding_box / ratio
        c.bounding_box[0] = np.maximum([0.0, 0.0], c.bounding_box[0])
        c.bounding_box[1] = np.minimum([1.0, 1.0], c.bounding_box[1])
      if relative_coord is False:
        c.bounding_box = c.bounding_box * [img.size]
    return candidates

  def get_input_tensor_shape(self):
      print(self.input_details[0]['shape'])
      return self.input_details[0]['shape']

  def DetectWithInputTensor(self, input_tensor, threshold=0.1, top_k=3):
    """Detects objects with raw input.

    This interface allows user to process image outside the engine for
    efficiency concern.

    Args:
      input_tensor: numpy.array represents the input tensor.
      threshold: float, threshold to filter results. Default value = 0.1.
      top_k: keep top k candidates if there are many candidates with score
        exceeds given threshold. By default we keep top 3.

    Returns:
      List of DetectionCandidate.

    Raises:
      ValueError: when input param is invalid.
    """
    if top_k <= 0:
      raise ValueError('top_k must be positive!')
    
    
    #_, raw_result = self.RunInference(input_tensor)
    print('input tensor', input_tensor)
    print('input tensor', input_tensor.shape)
    self.interpreter.set_tensor(self.input_details[0]['index'], np.reshape(input_tensor, self.get_input_tensor_shape()) )
    self.interpreter.invoke()
    
    
    result = []
    num_candidates = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
    scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
    labels = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
    boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    
    for i in range(int(round(num_candidates))):
        
      score = scores[i]
      #raw_result[self._tensor_start_index[2] + i]

      
      if score > threshold:
        label_id = int(round(labels[i]))
        if label_id in ALLOWED_LABELS:
            y1 = max(0.0, boxes[i, 0])
            x1 = max(0.0, boxes[i, 1])
            y2 = min(1.0, boxes[i, 2])
            x2 = min(1.0, boxes[i, 3])
            result.append(DetectionCandidate(label_id, score, x1, y1, x2, y2))
    result.sort(key=lambda x: -x.score)
    return result[:top_k]
