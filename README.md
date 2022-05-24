Assignment overview  
In modern industrial applications, sensors are used to observe key machine characteristics. These can help detect slight deviations and issues to avoid major system failures through targeted repairs while keeping maintenance costs under control.
Such sensors are critical in the operation of wind turbines. Due to fluctuating winds that can negatively impact the turbines and the high costs of maintenance, specifically for offshore turbines, sensor readings need to be reliably converted into an operating mode. That is, given the sensor readings over time, we want to know whether the turbine is operating correctly or whether one of several issues is present. If issues are detected (reliably), targeted efforts can be made to alleviate them before a major system failure occurs.
Your task will be to predict, as accurately as possible, the operating mode of a wind turbine based on the time series data from two sensors.
You find three pickle files as part of the assignment. After running import pickle, you can load the files into your notebook as follows:


The data are sensor readings and operating modes for 4,000 turbine runs. time_series_1 and time_series_2 are NumPy arrays of shape (4000,5000). Each observation corresponds to 5,000 records of the turbine over time by one of the two sensors (time_series_1 measures the pitch angle in each second of operation, and time_series_2 measures the generator torque). y is the operating mode for each of the 4,000 turbine runs (0 if the turbine is healthy, 1 if the generator torque is faulty, 2 if the pitch angle is faulty, and 3 if both are faulty). Note that the dataset is balanced in that each operating mode is represented equally often.
Throughout, you should use validation and test datasets consisting of 15% of the observations each. Make sure to use the same split of observations for all tasks.  
  
Task description  
1. Create a recurrent neural network in TensorFlow to predict the operating mode of a wind turbine based on the two time series from the sensors. Before any implementation, carefully consider what type of approach (sequence-to-vector, sequence-to-sequence, or encoder-decoder) is most sensible here and how you need to manipulate the data, given that you have two different time series for each observation.  
 
Then, make sure that you try out the different layers and elements discussed in class, such as SimpleRNN, LSTM and Conv1D - while a certain amount of trial and error will be necessary, it is recommended that you tune your network systematically. Make sure to record your final validation set accuracy.
Another tool for analyzing time-series data is convolutional neural networks with 2D convolutional layers. For this to work, time series need to be converted into “images” (matrices of numbers). The paper “Convolutional neural network fault classification based on time series analysis for benchmark wind turbine machine” by Rahimilarki, Gao, Jin, and Zhang (published 2022 in “Renewable Energy” and available through the City-library)
describes how two-dimensional CNNs can be applied to the problem at hand. Consider sections 4 and 5 that depict the process of converting one or multiple time series into “images” used within a CNN.  
  
2. In your own words, explain why the approach outlined here can help analyze time-series data and why it might outperform RNNs.
3. Convert the data for use with a CNN. In particular, following the approach outlined in Scenario 2 (section 5.3 of the paper) and summarized in Figure 18, convert the two time series corresponding to one wind turbine run into a single (100,100,1) array (i.e., a gray- scale image).
4. In TensorFlow, replicate the CNN with three convolutional layers displayed in Figure 12 and train it on your data. Make sure to record your final validation set accuracy.
5. Can you do better by adjusting the CNN? Be creative in your design choices (you might also consider pre-trained CNN architectures) and record your final validation set accuracy.
6. Compare the models you have created so far (both RNNs and CNNs) and make a selection (making sure to justify this). Train that model on a combined training and validation set and evaluate it on your test set.  
  
Hints  
• Start by creating minimum viable products. Only once everything runs should you go back to it and see how to improve your models. The performance of your models will matter for evaluation, but not as much as having a complete answer.
• Don’t worry about technical details on wind turbines, which are not required for creating your classification model.
• Keep in mind that you have two different time series for the same observation, which need to be combined. If you are stuck trying to combine them, start by creating a model that takes only one of them and try to extend from there.
Materials to submit
• A Jupyter notebook that allows recreating your solutions

• Your written answers, within the Jupyter notebook. Make sure to create numbered sections within your notebook corresponding to the task at hand
• The trained final model chosen in task 6, as an .h5-file
