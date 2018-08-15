## Writeup
---

**Vehicle Detection Project**


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat_1.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/HOG_search_result.png
[image9]: ./examples/SVM_search_results.png
[image10]: ./examples/bboxes_and_heat_2.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell (execution #8) of the IPython notebook `Vehicle-Detection.ipynb` function `get_hog_features`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of 

`orientations = 11`,
`pix_per_cell = 4`,
`cell_per_block = 1`,
`hog_channel = "ALL"`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

For searching best HOG parameters I used ParameterGrid with following grid:

```python
param_grid = {'colorspace':    ['HSV', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'],
              'orient' :       [9, 11],
              'pix_per_cell':  [4, 8],
              'cell_per_block':[1, 2],
              'hog_channel':   [0, 1, 2, 'ALL']}
```

There was 191 combination for HOG parameters. After search all results was sorted by accurancy:

![alt text][image8]

And the second best result 
```python
colorspace = 'YCrCb' 
orient = 11
pix_per_cell = 4
cell_per_block = 1
hog_channel = "ALL"
```
was chosen for further use,
because of a rather small prediction time in comparison with the best result(0.00582 vs 0.00139).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For training SVM classifier I used `GridSearchCV` with following parameters: 

```python
parameters = {'kernel':('linear', 'rbf'), 'C':[1,5,7,10]}
```
Results of search was following:
![alt text][image9]

For further use was selected next SVM parameters:
```
best parameters: {'C': 5, 'kernel': 'rbf'}
best score:      0.98262 (+/-0.00173)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search was done in function `find_cars` and `draw_car_boxes`.
Scale and window`s y value was selected manually with trying different combinations.
The best result was shown by next values: 
```python
    ystart = 400
    ystop = 500
    scale = 1.4

    ystart = 400
    ystop = 650
    scale = 2.2
```
Example of windows overlap is the next:
![alt text][image3]

It does not show the best result, but more widows search significantly increase search time. All test images was classified correctly. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

For optimization I used only two sliding windows search, described above.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Also I used `decision_function` function from `LinearSVC` with threshold `0.7`.I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Here are six frames with the resulting bounding boxes and their corresponding heatmaps:

![alt text][image5]
![alt text][image10]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem of current approach is very slow image processing time, in comparison to YoloV3 it is very slow. Another problem is high probability of false detection even with heat map filtering. In project video false datection by shadow of tree is very noticeable. Search could be modified for searching not from scratch, but from position in previous frame. It could help with slow processing time.

