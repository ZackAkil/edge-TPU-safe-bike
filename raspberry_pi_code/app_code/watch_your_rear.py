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

"""A demo to classify Raspberry Pi camera stream."""

import argparse
import io
import time

import numpy as np
import picamera

import board
import neopixel

from PIL import Image


LED_COUNT = 49

pixels = neopixel.NeoPixel(board.D18, LED_COUNT, auto_write=False, brightness=0.7)

CPU_MODEL_PATH = "models/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
TPU_MODEL_PATH = "models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"

WIDTH = 300
HEIGHT = 300

def get_camera_resize_shape(model_input_width, model_input_height):
    # becuase the pi-camera can only resize the image to multiplles of 16/32
    resize_width = int(model_input_width / 32) * 32
    resize_height = int(model_input_height / 16) * 16
    return (resize_width, resize_height)

resize_width, resize_height = get_camera_resize_shape(WIDTH, HEIGHT)

# LED strip utility functions

def clear_leds():
    pixels.fill((0, 0, 0))

def draw_leds(from_x, to_x, label=1):
    if label == 2:
        paint = (0, 200, 0)
    else:
        paint = (50, 0, 0)
    
    led_from = int(LED_COUNT*from_x)
    led_to = int((LED_COUNT)*to_x)
    for i in range(led_from, led_to):
        pixels[i] = paint
    
def show_leds():
    pixels.show()
    


from annotator.annotator import Annotator

# do you want the on-screen visualisation of what the camera is seeing?
DISPLAY = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--cpu", action="store_true", help="Use CPU")
    args = parser.parse_args()

    # use tflite model on CPU or tflite_edge model on TPU?
    if args.cpu:
        from edgetpu.detection.engine import TfLiteDetectionEngine
        engine = TfLiteDetectionEngine(CPU_MODEL_PATH)
    else:
        from edgetpu.detection.engine import DetectionEngine
        engine = DetectionEngine(TPU_MODEL_PATH)



    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 40
        camera.rotation = 90
        _, width, height, channels = engine.get_input_tensor_shape()
        print('reshaping to ', _, width, height, channels )
        if DISPLAY:
            camera.start_preview()
            annotator = Annotator(camera, dimensions=(640, 480))
        try:
            stream = io.BytesIO()
            print('imag shape', width, height)
            saved = False

            new_image = np.zeros((width,height,3),  dtype=np.uint8 )

            for foo in camera.capture_continuous(stream,
                                                 format='rgb',
                                                 use_video_port=True,
                                                 resize=(resize_width, resize_height)
                                                 ):
                stream.truncate()
                stream.seek(0)
                # get image pixels from camera
                input = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                # reshape to fit model input tensor shape
                new_image[:resize_width,:resize_height,:] = np.reshape(input,(resize_width, 
                                                                              resize_height,3))
                if 0:
                    if not saved:
                        print('trying to save', new_image.shape)
                        Image.fromarray(new_image).save('test_imag.png')
                        saved = True

                # actually get prediction from edge TPU, (and time how long it takes)
                start_ms = time.time()
                results = engine.DetectWithInputTensor(np.reshape(new_image, (300*300*3)) , top_k=10,  threshold=0.05)
                elapsed_ms = time.time() - start_ms
                
                clear_leds()
                
                if DISPLAY:
                    annotator.clear()
                    camera.annotate_text = "%.2fms" % (elapsed_ms*1000.0)

                if results:
                    for result in results:
                        # get bounding box
                        box = result.bounding_box
                    
                        if DISPLAY:
                            annotator.bounding_box((box[0][0]*640,
                                                box[0][1]*480,
                                                box[1][0]*640,
                                                box[1][1]*480))
                        
                        # draw on led strip
                        draw_leds(box[0][0], box[1][0], result.label_id)
                        
                    annotator.update()
                    show_leds()
                else:
                    # clear LEDs
                    draw_leds(0, 0)
                    show_leds()

        finally:
            if DISPLAY:
                camera.stop_preview()
            pass

if __name__ == '__main__':
    main()
