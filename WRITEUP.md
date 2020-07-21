# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers involves...

- By definition, Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

Some of the potential reasons for handling custom layers are...

- The Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files.

- The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. When your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error. 

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

- Accuracy of the pre-conversion model = moderate (less than post-conversion) and post-conversion model = Good

The size of the model pre- and post-conversion was...

- size of the fozen inference graph(.pb file) = 28Mb and size of the pos-conversion model xml+bin file = 27Mb

The inference time of the model pre- and post-conversion was...

- Inference time of the pre-conversion model:- Avg inference time: 128 ms

- Inference time of the post-conversion model:- Avg inference time: 43 ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

 - Now with COVID19, all places need to controll people amount inside restaurants, markets and    malls. With People Counter app you make this more easy. 
 - You can know what is the path to find a specific product;
 - Where is the best place to put you product. 
 - Where is the path most used . Etc.

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting: Lighting is most assential factor which affects to result of model. We need input image with lighting because model can't predict so accurately if input image is dark. So monitored place must have lights.

- Model accuracy: Deployed edge model must have high accuracy because deployed edge model works in real time if we have deployed low accuracy model then it would give faulty results which is no good for end users.

- Camera focal length: High focal length gives you focus on specific object and narrow angle image while Low focal length gives you the wider angle. Now It's totally depend upon end user's reuqirements that which type of camera is required. If end users want to monitor wider place than high focal length camera is better but model can extract less information about object's in picture so it can lower the accuracy. In compare if end users want to monitor very narrow place then they can use low focal length camera.

- Image size: Image size totally depend upon resolution of image. If image resolution is better then size will be larger. Model can gives better output or result if image resolution is better but for higher resolution image model can take more time to gives output than less resolution image and also take more memory. If end users have more memory and also can manage with some delay for accurate result then higher resoltuion means larger image can be use.

## How to Execute

- 1 Download model: 
  SSD_V1 : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  SSD_V2 : http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  
- 2 Extract to current dir: tar -xzvf <package.tar.gz>

- 3 Run model optimizer: 
  cd /opt/intel/openvino/deployment_tools/model_optimizer
  
  python mo_tf.py --input_model /home/workspace/ssd_v1/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json   --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_v1/pipeline.config --reverse_input_channels -o /home/workspace/ssd_v1
  
- 4 Execute:
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_v1/frozen_inference_graph.xml -l
  /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning 
  -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
