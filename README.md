# forest-fire

The work is devoted to the development of an open-source library of algorithms that can help in the real-time detection of forest fires on satellite images. Three algorithms have been developed the fire detection algorithm, the algorithm for calculating fire characteristics, and a prediction algorithm.

The fire detection algorithm enables in two steps to perform a search for fires. At first using threshold segmentation and watershed segmentation areas of interest are marked. Then the feature detector performs a search of fires in marked areas.

Using binary image analysis, the algorithm for calculates the characteristics selects a burned area and calculates its characteristics such as coordinates, area and etc.

The prediction algorithm compares several images taken at different times and calculates the estimated speed of fire, taking into account the wind rose.

The OpenCV and scikit-image libraries have been used to work with images. A wrapper for calling library methods was written in Python.
