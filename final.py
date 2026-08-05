import numpy as np
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from object_detection.utils import ops as utils_ops
from time import sleep
import smtplib
from email.message import EmailMessage
import imghdr
import os
from twilio.rest import Client
from time import sleep
import face_recognition
import winsound



def beep():
  frequency = 2500
  duration = 3000
  winsound.Beep(frequency, duration)

  



def face():
  # Get a reference to webcam #0 (the default one)
  video_capture = cv2.VideoCapture(0)
  print("camera enable")
  # Load a sample picture and learn how to recognize it.
  JOYCHRISTY_image = face_recognition.load_image_file("JOYCHRISTY.jpeg")
  JOYCHRISTY_face_encoding = face_recognition.face_encodings(JOYCHRISTY_image)[0]

  # Create arrays of known face encodings and their names
  known_face_encodings = [JOYCHRISTY_face_encoding]

  known_face_names = ["JOYCHRISTY"]

  # Initialize some variables
  face_locations = []
  face_encodings = []
  face_names = []
  process_this_frame = True
  name=""
  while True:
      # Grab a single frame of video
      ret, frame = video_capture.read()

      # Resize frame of video to 1/4 size for faster face recognition processing
      small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
      rgb_small_frame = small_frame[:, :, ::-1]

      # Only process every other frame of video to save time
      if process_this_frame:
          # Find all the faces and face encodings in the current frame of video
          face_locations = face_recognition.face_locations(rgb_small_frame)
          face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

          face_names = []
          for face_encoding in face_encodings:
              # See if the face is a match for the known face(s)
              matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
              name = "Unknown"

              # # If a match was found in known_face_encodings, just use the first one.
              # if True in matches:
              #     first_match_index = matches.index(True)
              #     name = known_face_names[first_match_index]

              # Or instead, use the known face with the smallest distance to the new face
              face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
              best_match_index = np.argmin(face_distances)
              if matches[best_match_index]:
                  name = known_face_names[best_match_index]

              face_names.append(name)

      process_this_frame = not process_this_frame


      # Display the results
      for (top, right, bottom, left), name in zip(face_locations, face_names):
          # Scale back up face locations since the frame we detected in was scaled to 1/4 size
          top *= 4
          right *= 4
          bottom *= 4
          left *= 4

          # Draw a box around the face
          cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

          # Draw a label with a name below the face
          cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

      # Display the resulting image
      cv2.imshow('Video', frame)
      if(name=="JOYCHRISTY"):
          print("JOYCHRISTY FACE DETECTED....")
          
      # Hit 'q' on the keyboard to quit!
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # Release handle to the webcam
  video_capture.release()
  cv2.destroyAllWindows()









sys.path.append("..")


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    if output_dict['detection_classes'][0] == 1 and  output_dict['detection_scores'][0] > 0.80:
      print('Hammer')
##      beep()
      face()
      
      
    if output_dict['detection_classes'][0] == 2 and  output_dict['detection_scores'][0] > 0.70 :
      print('Helmet')
      #beep
      #face()
      
    return output_dict
    
    

import cv2
cap = cv2.VideoCapture(0)

try:
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)

                while True:
                    ret, image_np = cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph)
                    # Visualization of the results of a detection.
                    
                    if output_dict['detection_classes'][0] == 1 and  output_dict['detection_scores'][0] > 0.90:
                      print('theft detected')
                      
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          output_dict['detection_boxes'],
                          output_dict['detection_classes'],
                          output_dict['detection_scores'],
                          category_index,
                          instance_masks=output_dict.get('detection_masks'),
                          use_normalized_coordinates=True,
                          line_thickness=8)
                    if output_dict['detection_classes'][0] == 2 and  output_dict['detection_scores'][0] > 0.90:
                      print('Helmat')
                      
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          output_dict['detection_boxes'],
                          output_dict['detection_classes'],
                          output_dict['detection_scores'],
                          category_index,
                          instance_masks=output_dict.get('detection_masks'),
                          use_normalized_coordinates=True,
                          line_thickness=8)
                    cv2.imshow('Frame', cv2.resize(image_np,(800,600)))
                    cv2.imwrite('capture.jpg',image_np)
                    if cv2.waitKey(1) == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
except Exception as e:
    print(e)
    cap.release()
