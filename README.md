> **Depth map from stereo camera**
> 
>
> **1. Calibration**\
> a) Features on the two images are found using SIFT detector. Then,\
> cv2.Bfmatcher is used to match the extracted features, and then this
> list of all matches is sorted based on distance parameter of Dmatch
> object. Out of all the matches, first n matches are taken which will
> closely resemble the matching features correctly. However, this match
> list will also contain some incorrectly matched features.

![](vertopal_fc3120865ac9415693acbe037064620c/media/image1.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image2.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image3.png)

> b\) RANSAC: To eliminate incorrectly matched features, we use RANSAC.
>
> First, 8 random indices are randomly selected, and the matches at
> these indices are found. Then, the points corresponding to these 8
> matches in the respective images are used to find the fundamental
> matrix using the eightpoint algorithm. We run RANSAC for 1000
> iterations, and calculate the inlier every run, and select the matches
> with most inliers and save the fundamental matrix formed using these 8
> matches.

![](vertopal_fc3120865ac9415693acbe037064620c/media/image4.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image5.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image6.png)

> c\) Essential matrix: Using the fundamental matrix and camera
> matrices, we calculate the essential matrix:\
> E = K2.T . F . K1
>
> d\) Camera pose: With essential matrix, we find the four possible
> camera poses using

![](vertopal_fc3120865ac9415693acbe037064620c/media/image7.png)

> where U, V are found using the SVD of E, and W is

![](vertopal_fc3120865ac9415693acbe037064620c/media/image8.png)

> e\) Cheirality condition: However, the correct camera pose is one
> which has all the Z values of triangulation as positive. Since there
> can be noise, there can be some Z values as negative in the
> triangulated points. Hence, we find the pose with maximum Z value.
>
> **2. Rectification**\
> Using the points in the two images and the fundamental matrix,\
> homographies are found for the two images which warp them to a common
> plane. This is done using cv2.stereoRectifyUncalibrated( ).
> Triangulated points are also perspective-transformed accordingly. We
> use

cv2.warpPerspective and cv2.perspectiveTransform( ). The fundamental
matrix is also rectified using\
*F = H2_inv . F . H1_inv*

Then, we find the epipolar lines for both images using image points and
the rectified fundamental matrix using cv2.computeCorrespondEpilines( ).
The lines in both images are parallel.

**3. Correspondence**\
Along each epipolar line, we find corresponding blocks of images between
the two images. For this, a search range is defined, 50 in this
implementation, and the block size is selected which will define the
size of block to be matched. The disparity maps will get smoother as we
increase this block size, but will contain less accurate disparity.
Along each epipolar line, a window is defined for the second image and
SSD is used to determine best match block in the second image. Then, we
update the pixels of disparity map (around which the block was centered)
with the absolute difference of the\
corresponding blocks. This process consumes most amount of time. Hence,
the input images were scaled down by a factor of 4 after loading them,
so as to save computation time.

**4. Depth map**\
The depth map is calculated using

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image9.png)

**Results**\
**1. Curule dataset**

![](vertopal_fc3120865ac9415693acbe037064620c/media/image10.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image11.png)
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image12.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image13.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image14.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image15.png)

Disparity map: gray(above) and heat(below)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image16.png)
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image17.png)

Depth map: gray(above) and heat(below)

> **2. Octagon dataset**

![](vertopal_fc3120865ac9415693acbe037064620c/media/image18.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image19.png)
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image20.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image21.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image22.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image23.png)

Disparity map: gray(left) and heat(right)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image24.png)
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image25.png)

Depth map: gray(left) and heat(right)

> **3. Pendulum dataset**
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image26.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image27.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image28.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image29.png)

![](vertopal_fc3120865ac9415693acbe037064620c/media/image30.png)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image31.png)

Disparity map: gray(above) and heat(below)

> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image32.png)
>
> ![](vertopal_fc3120865ac9415693acbe037064620c/media/image33.png)

Depth map: gray(above) and heat(below)
