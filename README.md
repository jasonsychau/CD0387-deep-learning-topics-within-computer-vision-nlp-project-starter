# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

Training jobs screencapture is trainin_jobs.png. The logs are in batch-size_10_lr_0_09386252244689186.png and batch-size_10_lr_0_09727184909755177.png. 

I chose the resnet18 model since I have experience with it from exercises (and it is pretrained for shorter training time). I used parameter ranges lr 0.9 to 1.0 and batch size 10 and 100. After training, I found that loss increased after two epoches. I guess that the optimizer went to far with the gradient and learning rate, so I reverted the model to lr 0.001 and batch size 32.

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
I used the debugger and profiler to plot the loss. The plot was showing that loss was increasing for my models.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

I found that loss was increasing: the learning rate was too large.

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The deployed endpoint is accepting bytes for an image, but the butes should be saved from and image. This is implemented in train_and_deploy.ipynb with the Image and io.BytesIO modules.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

File name is endpoint.png.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
