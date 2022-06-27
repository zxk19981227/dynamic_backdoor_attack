# Input-aware Dynnamic Backdoor Attack

This is the implementations for Input-aware Dynamic Backdoor Attack for natural language processing.

To run this code, you should install the **apex**(https://github.com/NVIDIA/apex) at first. The pytorch vision is
1.11.0+cu11.3. Other dependences are shown in the requirements.txt. The UNILM model is from https://github.com/microsoft/unilm/tree/master/unilm-v1.

The pre-trained language model should be download as the unilm/readme.md shows.

To train our model, you can first download the pre-trained model and install pytorch, apex. Then run the following codes:

`bash setup.sh
 pip install -r requirements.txt
python train.py --dataset sst`

to train the Standford Sentiment Classification task.