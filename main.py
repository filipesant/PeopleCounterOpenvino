"""People Counter
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
import time
import socket
import json
from argparse import ArgumentParser
from inference import Network
import cv2
import paho.mqtt.client as mqtt


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def draw_boxes(frame, result, width, height, prob_threshold, hist):
    """
    Method for drawing bounding boxes
    """
    counter = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            counter = counter + 1

    if counter > 0:
        hist = 5
    elif (counter == 0) and (hist > 0):
        counter = 1
        hist += -1

    return frame, counter, hist

def connect_mqtt():
    """
    Connect to the MQTT client
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video
    """
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Initial Parameters
    single_image_mode = False
    start_time = time.time()
    total_count = 0
    last_count = 0
    cur_request_id = 0
    num_requests = 20
    hist = -1

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, num_requests, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### Loop until stream is over ###
    while cap.isOpened():

        ### Reading from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-processing the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Detect Time Start
        infer_start = time.time()

        ### Starting asynchronous inference for specified request ###
        infer_network.exec_net(cur_request_id, p_frame)

        ### Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            # Updating Detect time
            detect_time = time.time() - infer_start

            ### Getting the results of the inference request ###
            result = infer_network.get_output(cur_request_id)

            ### Getting Inference Time
            infer_time = "Inference Time: {:.3f}ms".format(detect_time * 1000)

            ### Extract any desired stats from the results ###
            # Draw Bounding Boxes
            frame, current_count, hist = draw_boxes(frame, result, width, height, prob_threshold, hist)

            # Get a writen text on the video
            cv2.putText(frame, infer_time, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

            ### Calculate and send relevant information on ###
            if current_count > last_count:
                start_time = time.time()
                # current_count, total_count and duration to the MQTT server
                total_count = total_count + current_count - last_count
                # Topic "person": keys of "count" and "total"
                client.publish("person", json.dumps({"total": total_count}))

            ### Topic "person/duration": key of "duration" ###
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count, "total": total_count}))
            last_count = current_count
            cur_request_id += 1
            if cur_request_id == num_requests:
                cur_request_id = 0

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if key_pressed == 27:
            break

        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)


    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT
    client.disconnect()


def main():
    """
    Loading the network and parsing the output
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
