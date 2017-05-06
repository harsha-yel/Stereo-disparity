Obtain Stereo-vision disparity map using feature desriptors-SIFT, ORB
### Matching descriptors	
	Matching algorithm for SIFT: Flann based kd-tree 
	Matching algorithm for ORB: Flann based Locality sensitive hashing

The disparity, disparity map, RMS, Bad pixel percent calculations are implemented.

We compute disparity map for leftimage only.
	
	Example : Input data: Cones
	 	  Output Data: cones_sparse or cones_dense	
	Each Input folder contains 4 files:
			1. Left stereo image 			 -   left.png
			2)Right stereo image			 -   right.png
			3)disparity Ground truth for Left Image	 -   left_gt.png
			3)disparity Ground truth for Right Image -   right_gt.png
	Each output folder contains 4 images:
			1)disparity map obtained by SIFT 	    - disparity_SIFT.png
			2)disparity map obtained by ORB  	    - disparity_ORB.png
			3)Good matches between keypoints using SIFT - GoodMatches_SIFT.png
			4)Good matches between keypoints using ORB  - GoodMatches_ORB.png

Performed the experiment on 3 datasets of middlebury stereovision----> Cones, Teddy, Art
	
### performed both sparse and dense disparities
	Default: Sparse
	To do dense disparities: Uncomment the lines 161,162 and comment the lines 158,159 in main()	
	Do not run dense disparity on "Art" dataset. Due to its higher resolution, takes a lot of time(in hours) 
	and is not recordable in the lower-medium PC's. 
### Bad pixel percentage default threshold=1,2 and 5
Running the code by all defaults produces cones_out_sparse, Teddy_out_sparse, Art_out_sparse

### Folders:

	1)code folder   : contains the project files and source code.
	2)Input Folder  : Contains the three datasets which are used for input
	3)Output Folder : Contains the sample outputs(already generated by student) for the input datasets		 	
	When the code is executed new output folders are created within the Output folder.
	