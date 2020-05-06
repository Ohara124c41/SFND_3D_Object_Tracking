# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks:
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences.
2. Second, you will compute the TTC based on Lidar measurements.
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches.
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Tasks
- [x] FP.0 Final Report
- [x] FP.1 Match 3D Objects
- [x] FP.2 Compute Lidar-based TTC
- [x] FP.3 Associate Keypoint Correspondences with Bounding Boxes
- [x] FP.4 Compute Camera-based TTC
- [x] FP.5 Performance Evaluation 1
- [x] FP.6 Performance Evaluation 2


### FP.1 Match 3D Objects
>Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    std::multimap<int, int> IDPrevCurr;
    for (auto match : matches) {
        cv::KeyPoint kpt = prevFrame.keypoints[match.queryIdx];
        int IDPrevious = -1;
        int IDCurrent = -1;
        for (auto &bb : prevFrame.boundingBoxes) if (bb.roi.contains(kpt.pt)) IDPrevious = bb.boxID;
        for (auto &bb : currFrame.boundingBoxes) if (bb.roi.contains(kpt.pt)) IDCurrent = bb.boxID;
        if (IDPrevious != -1 && IDCurrent != -1) IDPrevCurr.emplace(IDPrevious, IDCurrent);
    }
    for (auto const &bb : prevFrame.boundingBoxes) {
        auto range = IDPrevCurr.equal_range(bb.boxID);
        std::map<int, int> ReoccuringIDCurrent;

        for (auto it = range.first; it != range.second; it++) {
            auto subReoccuringID = ReoccuringIDCurrent.find(it->second);
            if (subReoccuringID != ReoccuringIDCurrent.end()) subReoccuringID->second++;
            else ReoccuringIDCurrent.emplace(std::make_pair(it->second, 1));
        }
        int maxCurrentCount = 0;
        int peakIDCurrent = -1;

        for (auto it = ReoccuringIDCurrent.begin(); it != ReoccuringIDCurrent.end(); it++) {
            if (maxCurrentCount < it->second) {
                peakIDCurrent = it->first;
                maxCurrentCount = it->second;
            }
        }
        if (peakIDCurrent != -1) bbBestMatches.emplace(std::make_pair(bb.boxID, peakIDCurrent));
    }
}

```

![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/classification.PNG?raw=true)
### FP.2 Compute Lidar-based TTC

>Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.



```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    vector<double> xPrevious, xCurrent;
    for (auto p : lidarPointsPrev) xPrevious.push_back(p.x);
    for (auto p : lidarPointsCurr) xCurrent.push_back(p.x);
    TTC = (xPrevious.size() && xCurrent.size()) ? median(xCurrent) / (median(xPrevious) - median(xCurrent)) / frameRate : NAN;
}
```
![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/kptAKAZE.PNG?raw=true)
### FP.3 Associate Keypoint Correspondences with Bounding Boxes
>Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
    std::vector<double> distances;
    std::vector<cv::DMatch> keypointInRoI;
    for (auto kptMatch: kptMatches) {
        cv::Point pPrevious = kptsPrev[kptMatch.queryIdx].pt;
        cv::Point pCurrent = kptsCurr[kptMatch.trainIdx].pt;
        if (boundingBox.roi.contains(pCurrent)) {
            distances.push_back(cv::norm(pCurrent - pPrevious));
            keypointInRoI.push_back(kptMatch);
        }
    }
    double mu = median(distances);
    auto mnmx = std::minmax_element(distances.begin(), distances.end());
    double range = mnmx.second - mnmx.first;
    for (int i = 0; i < distances.size(); ++i)
        if (fabs(distances[i] - mu) < range * .8)
            boundingBox.kptMatches.push_back(keypointInRoI[i]);
}

```


### FP.4 Compute Camera-based TTC
>Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

 ```c++
 void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                       std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
     if(kptMatches.size()==0) return;
     vector<double> distanceRatios;
     for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
     {
       	// Outer KeyPoint Loop
         cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
         cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

         for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
         {
           	// Inner KeyPoint Loop
             double minDistance = 100.0;

             cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
             cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

             double currDistance = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
             double prevDistance = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

             if (prevDistance > std::numeric_limits<double>::epsilon() && currDistance >= minDistance)
             {
               	// Anti-NullDiv
                 double distanceRatio = currDistance / prevDistance;
                 distanceRatios.push_back(distanceRatio);
             }
         }
     }

     if (distanceRatios.size() == 0)
     {
         TTC = NAN;
         return;
     }

     // Retrieve camera-based TTC from distance ratios
     double aveDistanceRatio = std::accumulate(distanceRatios.begin(), distanceRatios.end(), 0.0) / distanceRatios.size();
     double deltaPeriod = 1 / frameRate;
     TTC = -deltaPeriod / (1 - aveDistanceRatio);
 }
```

### FP.5 Performance Evaluation 1
>Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.


The next image demonstrates a TCC error with the LiDAR detector in which the result is `inf`. As such, the detector has not properly identified the object. LiDAR is susceptible to errors when light is reflected incorrectly and the round-trip time has noise/deviations. Furthermore, the classifier has focused on the transportation vehicle frame on the right side of the image. The classifier is also confused regarding the passenger's rear tail light.

![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/errLiDAR.png?raw=true)

The following image shows the results of the LiDAR TCC [s] for 18 images. As can be seen, the lowest valley occurs immediately before the peak value. This indicates that either the sensor or the predictor had unwanted noise/variance in the channel as the image timestamped at the exact time has a value of `TCC LiDAR: 13.1241118s`.

![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/chartLiDAR.PNG?raw=true)


### FP.6 Performance Evaluation 2

>Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

In the image example below, the TCC camera value is reported as `-761.58s`, which is impossible. This might have been caused by deviations in the egomotion prediction. Furthermore, cameras have difficulty in properly identifying objects directly in front of them. As such, sensor fusion is critical for safety-critical systems.
![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/errCamera.png?raw=true)


Next, benchmarking was conducted for all combinations. The images below demonstrates three combinations that have issues in the average TCC for cameras. These combinations are `ORB+ORB`, `HARRIS+BRIEF`, and `AKAZE+BRISK`.


![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/radioCameraAll.PNG?raw=true)

Pruning out these three detector + descriptor combinations shows a more robust distribution of average behavior for camera TCCs. These are plotted with the standard deviations.

![image](https://github.com/Ohara124c41/SFND_3D_Object_Tracking/blob/master/images/radioCam.PNG?raw=true)

As can be seen, the `FAST+BRISK` has one of the lowest average TCC values and has the lowest standard deviation. `SHITOMASI+ORB` and `SHITOMASI+SIFT` also perform well.

Cross-referencing the *KeyPoint Detection* results from the of the [Midterm Project](https://github.com/Ohara124c41/SFND_2D_Feature_Tracking), results in the following list (considering top three previous detector+descriptor combinations and top three performers for this project):

Detector+Descriptor | Average Camera TCC [s]|Standard Deviation|TTC Difference [Camera - LiDAR]
--------------------|-----------------------------|----|----|
`FAST+BRIEF`|13.9344|5.6357|12.6148|
`FAST+BRISK`|**12.1659**|**1.09186**|10.8466|
`FAST+ORB`|12.7135|1.7621|11.3939|
`FAST+SIFT`|12.9020|1.9617|11.5824|
`SHI-TOMASI+SIFT`|**11.8644**|**1.2874**|10.5448|
`SHI-TOMASI+ORB`|**11.9235**|**1.5117**|10.6039|




The 2D-Feature Tracking performance results are as follows:

| Detector + Descriptor | Num. KP | Match KP |Time [ms]|Match KP /Num. KP|Num. KP/Max. KP|Best Time/Actual Time |Score*|
|:--------:| :-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| `FAST+BRIEF`|**4094**|2178|49.1297|0.532|1|0.985|**52.410**
| `FAST+BRISK`|**4094**|1832|69.2639|0.448|1|0.699|31.306
| `FAST+ORB`|**4094**|2061|**48.4005**|0.503|1|**1.000**|**50.300**
| `FAST+SIFT`|**4094**|**2782**|52.0133|**0.680**|1|0.931|**63.277**
| `SHI-TOMASI+ORB`|1179|768|167.014|0.651|0.288|0.290|5.433
| `SHI-TOMASI+SIFT`|1179|927|155.641|**0.786^**|0.288|0.311|7.039

**^** High accuracy but approximately 1/4th of total detect KPs; potentially better for low memory footprint.


## Detector + Descriptor Selection

As a result of the tables above, the following combinations have been selected prioritizing response time, accuracy, total keypoints detected, and minimal errors:


| Detector + Descriptor | Motivation |
|:--------:| :-------------|
| FAST+SIFT| Low Std. Err., **Highest Acc.**, Fast RT, Good TCC RT, **Highest Tradeoff Space Score** |
| FAST+ORB| Low Std. Err., High Acc., **Fastest RT**, Good TCC RT |
| FAST+BRISK| **Lowest Std. Err.**, High Acc., Good RT, **Excellent TCC RT** |
