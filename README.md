# UltrasoundImgRecon

This study systematically evaluates different CNN architectures, including U-Net, MobileNet, and CNN for adaptive ultrasound image reconstruction.The implementation can be found in the code snippet titled usimgrecon.py

The code can be easily implemented and hosted on Google Colab that provides hardware support for training the network. With a hardware accelerator (GPU) the speed of execution is manifold greater.

The implementation pipleine is given below, where the neural network can be simple CNN,  obilenet and U-Net.
![image](https://github.com/user-attachments/assets/6d0c16aa-aa94-4bfd-bdfb-2f01c01461ac)

This study presented a comparative analysis of CNN, MobileNet, and U-Net architectures for image reconstruction tasks using MAE, MSE, and SSIM metrics. The U-Net model demonstrated superior performance, achieving the lowest error rates and highest structural similarity, making it highly effective for high-fidelity tasks.

In this study, the datasets and experimental setup were carefully selected to evaluate the performance of various deep learning models for ultrasound image reconstruction. For training, the Time-of-Flight (ToF) corrected ultrasound RF data obtained from [1] was utilized. The data for training is obtained from the Verasonics research ultrasound machine (Verasonics Vantage 128 and L11-5v transducer, Verasonics, Kirkland, WA, USA) with center frequency of 7.6 MHz and sampling frequency of 31.25 MHz. This dataset contains 100 frames of arm and finger scans from a healthy subject, providing a robust basis for model learning.
Testing was conducted using the publicly available PICMUS dataset, which includes ultrasound RF data of carotid tissue, ensuring a clear distinction between training and test datasets. This separation, where the training and test data originate from entirely different sources, eliminates overlap and ensures an unbiased evaluation of model performance.

[1]Mathews, Roshan P., and Mahesh Raveendranatha Panicker. "Towards fast region adaptive ultrasound beamformer for plane wave imaging using convolutional neural networks." In 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), pp. 2910-2913. IEEE, 2021.
