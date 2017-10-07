## Camera Calibration

Each chessboard has certain number of corners in both the *image* and *3D object* corners. The *objects* have 3D coordinates, since they are real world coordinates. The *image* corners are using corner detection (`findChessboardCorners` from openCV). If the function finds corners, they go from left to right and then down.

We use the function `drawChessboardCorners` to draw *on the image* the corners. We then find the camera parameters in the form of a matrix (mtx), and then undistort it.

## Perspective Transform

Perspective come about that things that are further away appear smaller. That's why the lane looks smaller the further away. Mathematically, in real world coordinates, the smaller the *z* coordinate, the smaller the appearance, and the further it is away.

We can adjust for this with a *perspective transform*, and thereby creating a *birds eye view transform*. Applications can actually map the 3d points to a map using this technique.

Using the function `getPerspectiveTransform`, I can take corresponding source and destination and find the perspective transformation, linearly. To reproject, I can use the function by switching source and destination. To warp the image, I can use `warpPerspective`.


