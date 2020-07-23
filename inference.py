#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        """
        Initializing class variables
        """
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):
        """
        Load the model
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        if not plugin:
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        self.network = IENetwork(model=model_xml, weights=model_bin)

        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        self.exec_network = self.plugin.load_network(self.network, num_requests=num_requests, device_name=device)
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return self.plugin

    def get_input_shape(self):
        """
        Returning the shape of the input layer
        """
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, req_id, image):
        """
        Starting an asynchronous request
        """
        self.infer_request = self.exec_network.start_async(request_id=req_id, inputs={self.input_blob: image})
        return self.exec_network

    def wait(self, req_id):
        """
        Waiting for the request to be complete
        """
        status = self.exec_network.requests[req_id].wait(-1)
        return status

    def get_output(self, req_id):
        """
        Returning the output results
        """
        return self.exec_network.requests[req_id].outputs[self.output_blob]
