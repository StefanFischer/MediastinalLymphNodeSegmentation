# MediastinalLymphNodeSegmentation Algorithm

Challenge Submission of the CompAI team for the MICCAI Lymph Node Quantification Challenge

# Challenge Setting

Accurate lymph node size estimation is critical for staging cancer patients, initial therapeutic management, and in longitudinal scans, assessing response to therapy. Current standard practice for quantifying lymph node size is based on a variety of criteria that use unidirectional or bidirectional measurements on just one or a few nodes, typically on just one axial slice. But humans have hundreds of lymph nodes, any number of which may be enlarged to various degrees due to disease or immune response. While a normal lymph node may be approximately 5 mm in diameter, a diseased lymph node may be several cm in diameter. The mediastinum, the anatomical area between the lungs and around the heart, may contain ten or more lymph nodes, often with three or more enlarged greater than 1 cm. Accurate segmentation in 3D would provide more information to evaluate lymph node disease. Full 3D segmentation of all abnormal lymph nodes in a scan promises to provide better sensitivity to detect volumetric changes indicating response to therapy. 

<br>

The LNQ2023 Challenge training dataset will consist of a unique set of high-quality pixel-level annotations of one or more clinically relevant lymph nodes in a dataset part of cancer clinical trials. The goal will be to segment all lymph nodes larger than 1 cm in diameter in the mediastinum. Participants will be provided with a subset of cases that are partially annotated (i.e. one node out of five), and evaluation of the algorithms will be performed on a distinct dataset that is fully annotated (i.e. all clinically relevant nodes).


## Using the model via the docker container

this GitHub repository already includes the model weights, which were used for the final submission for the MICCAI LNQ2023 challenge. If you want to use the model you can use test.sh script that will build a docker container and run inference on the image at location './test/images/mediastinal-ct/*.nii.gz' (only one image per run).

We offer the model weights and inference process of our challenge submission and Model 7 of our technical report. Final Performance on the LNQ2023 test set can be found in our Technical Report.

## Using the model via Grand Challenge Website

You can also use the algorithms docker container hosted at https://grand-challenge.org/algorithms/mediastinallymphnodesegmentation/



# Training of own Model
If you want to train a model on your own data, you just need to install and use nnUNet.




