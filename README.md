![maxresdefault](https://user-images.githubusercontent.com/61338647/170028290-c2b58093-8cf4-4ee9-b08b-77d0cc21e3f8.jpg)
## Assignment overview.  
In modern industrial applications, sensors are used to observe key machine characteristics. These can help detect slight deviations and issues to avoid major system failures through targeted repairs while keeping maintenance costs under control.
Such sensors are critical in the operation of wind turbines. Due to fluctuating winds that can negatively impact the turbines and the high costs of maintenance, specifically for offshore turbines, sensor readings need to be reliably converted into an operating mode. That is, given the sensor readings over time, we want to know whether the turbine is operating correctly or whether one of several issues is present. If issues are detected (reliably), targeted efforts can be made to alleviate them before a major system failure occurs.
Your task will be to predict, as accurately as possible, the operating mode of a wind turbine based on the time series data from two sensors.
You find three pickle files as part of the assignment. After running import pickle, you can load the files into your notebook as follows:


The data are sensor readings and operating modes for 4,000 turbine runs. time_series_1 and time_series_2 are NumPy arrays of shape (4000,5000). Each observation corresponds to 5,000 records of the turbine over time by one of the two sensors (time_series_1 measures the pitch angle in each second of operation, and time_series_2 measures the generator torque). y is the operating mode for each of the 4,000 turbine runs (0 if the turbine is healthy, 1 if the generator torque is faulty, 2 if the pitch angle is faulty, and 3 if both are faulty). Note that the dataset is balanced in that each operating mode is represented equally often.
Throughout, you should use validation and test datasets consisting of 15% of the observations each. Make sure to use the same split of observations for all tasks.  
  

