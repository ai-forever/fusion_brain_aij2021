# Fusion Brain Challenge

The current task suggests developing a single multitask model, which would successfully complete such multimodality subtasks as **Code2code translation (С2С), Zero-shot object detection (OD), Handwritten Text Recognition (HTR), Visual Question Answering (VQA)** and be able to break an integral metric of baseline proposed by the Arranger, as well as metrics for each subtask.

## General solution format

Participants should create a zip archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container. Solutions may be run with the use of any image available for downloading from DockerHub. If necessary, you can prepare your own image, add necessary software and libraries to it (see the manual on creating Docker images); to use it, you will need to publish it on DockerHub.

The archive root must contain the metadata.json file containing the following:
```
{
    "image": "sberbank/fb-python",
    "entry_point": "python run.py $PATH_INPUT $PATH_OUTPUT"
}
```
Where ```image``` is a field with the docker image name, in which the solution will be run, ```entry_point``` is a command that runs the solution. For solution, the archive root will be the current directory. During the run, the ```DATASETS_PATH``` environment variable shall contain the route to relevant open datasets accessible from the container with the solution.

An argument accepted by the script for model inference should be represented by a path to the folder with the content to be used for prediction. Let us assume that the argument shall be represented by the ```fusion_brain``` folder. Names of ```fusion_brain``` subfolders shall correspond to names of subtasks to be completed by the single model. Each subfolder (HTR, OD, VQA, C2C) shall have the content needed for predictions. Below we discuss the structure of this content for each subtask in more detail.

The data structure is as follows:

* fusion_brain
  * HTR
    * images 
  * OD
    * images
    * requests.json
  * VQA
    * images
    * questions.json
  * C2C 
    * requests.json

The single model should generate predictions for each subtask in the ```prediction_{TASK_NAME}.json``` format, i.e. there should be four files after the model inference: ```prediction_HTR.json, prediction_OD.json, prediction_VQA.json, prediction_C2C.json```. These files should be located in the root of uploaded file with the solution.

After that, correct answers in the ```true_{TASK_NAME}.json``` format shall be added to the container and a script for calculation of metrics for each subtask shall be run. The final metric shall be calculated as a sum of metrics for each subtask (see below).


