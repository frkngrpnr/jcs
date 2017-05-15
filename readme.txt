This folder contains the code for our entry in the Apparent Personality Analysis and Job Candidate Screening Coopetition Workshop @CVPR 2017.

%% Notes
1) This code runs in MATLAB, and tested in a Linux system, if you want to run it on a Windows-based system instead, please change line 17 of main.m accordingly, i.e. bss = '\';

2) Since the features are too big to upload to GitHub, they are placed in a separate (Dropbox) folder. You can find them on the following  link:
 https://www.dropbox.com/sh/ciem9jt79jfbt28/AACg5e1MVQvQa_LjVc3XRcisa?dl=0 
Please make sure that the .mat files are placed under the path ./data/features.

3) The code produces the test set estimations using the development (training+validation) set. Note that the features extracted contain both training, validation and test set instances. In the given features, the test set instances are separated to ease memory/computations.

4) To avoid conflict of interest, we didn't include third party tools and pre-trained models used for face detection and (partly) in feature extraction. We, however, indicated the necessary pointers and links for these external resources
For face alignment, you need the IntraFace library. For feature extraction, you need VLFeat (available at http://www.vlfeat.org/download.html), MatConvNet (available at http://www.vlfeat.org/matconvnet/) and OpenSmile (available at http://audeering.com/research/opensmile/) libraries installed. For audio feature extraction, we use the IS13_ComParE.conf file.

5) Our FER 2013 fine-tuned VGG-Face network model can be accessed from: https://drive.google.com/open?id=0B2KpGwIOmPOieURkZE5aX3VIaFE
If you use this fine-tuned network, please also cite:
H. Kaya, F. Gürpınar, A. A. Salah, Video-Based Emotion Recognition in the Wild using Deep Transfer Learning and Score Fusion, Image and Vision Computing, Available online 4 February 2017, http://dx.doi.org/10.1016/j.imavis.2017.01.012.

6) If the corresponding flag is set, the test set predictions will be written to predictions.pkl reading the template file prediction_template_test.pkl (provided by the organizers) and predictions.csv that we create from final predicitons. 


Happy hacking!
Heysem, Furkan & Albert
