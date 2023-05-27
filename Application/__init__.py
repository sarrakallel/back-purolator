# import numpy as np

# import depthai as dai

# import cv2 as cv

 

# class HeightCamera:

 

#     """

#     This class manages an OAK camera, and computes depth information from its input. It assuems it is looking down at an object on a flat floor, and can compute distances to the object.

 

#     To compute depth data, the class needs to record multiple frames of input and filter them. It can either manage its own capture intervals or let the caller do it. When managing its own capture intervals, depth queries may take longer, as it needs time to capture data. If this is an issue, then set manage_own_capture to False, and periodically call capture_depth_frame.

#     """

 

#     # Everything will be in VGA resolution

#     WIDTH = 640

#     HEIGHT = 480

 

#     def __init__(self,

#                  manage_own_capture=True,

#                  median_filter_frames=5,

#                  fixed_floor_distance=None,

#                  object_noise_threshold=50,

#                  spotlight_radius=75,

#                  fps=15):

#         """

#         manage_own_capture: When True, the class will record frames of data whenever distance methods are called. When False, the caller must make calls to capture_depth_frame() periodically.

#         median_filter_frames: How long an interval the algorithm should consider when computing depths.

#         fixed_floor_distance: Set this to hardcode a distance to the floor in mm.

#         object_noise_threshold: Minimum height an object detection must be to register in mm. Needs to be larger for higher mount points of the camera.

#         spotlight_radius: configures the bias region, everything inside this radius from the centre of the image will be considered for minimal distance calculations, and that radius is most strongly considered for mean distance calculations

#         """

#         self.manage_own_capture = manage_own_capture

#         self.median_filter_frames = median_filter_frames

#         self.median_filter_buffer = []

#         self.fixed_floor_distance = fixed_floor_distance

#         self.object_noise_threshold = object_noise_threshold

 

#         mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

 

#         # Bias function for sampling, does not guarantee all the values sum to 1

#         # (would be a pointless optimization since different images will mask out different parts of the function)

#         # Use an exponential falloff about the centre

 

#         cx = HeightCamera.WIDTH // 2

#         cy = HeightCamera.HEIGHT // 2

#         # the alpha value stretches out the region of bias

#         # alpha = radius at which pixel contribution is halved from centre of image

#         alpha = 1.0 / spotlight_radius

#         self.bias = np.array(

#             [np.array(

#                 [(2 / (1 + alpha**2*((x-cx)**2 + (y-cy)**2)))  # 2.0 at centre, 1.0 at r=alpha, decays outward

#                  for x in range(HeightCamera.WIDTH)])

#              for y in range(HeightCamera.HEIGHT)])

 

#         # print(self.bias[cy, cx]) # 2.0

#         # print(self.bias[cy + int(1/alpha), cx]) # 1.0

 

#         # Make a spotlight for minimum distance checks

#         _, self.clipped_bias = cv.threshold(

#             self.bias, 1.0, 1.0, cv.THRESH_BINARY)

 

#         # DepthAI manager classes

#         self.pipeline = dai.Pipeline()

#         self.device = dai.Device()

 

#         # Pipeline nodes

#         cam_rgb = self.pipeline.create(dai.node.ColorCamera)

#         cam_left = self.pipeline.create(dai.node.MonoCamera)

#         cam_right = self.pipeline.create(dai.node.MonoCamera)

#         node_stereo = self.pipeline.create(dai.node.StereoDepth)

#         out_rgb = self.pipeline.create(dai.node.XLinkOut)

#         out_depth = self.pipeline.create(dai.node.XLinkOut)

 

#         # Configure cameras

#         # RGB

#         cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

#         cam_rgb.setFps(fps)

#         cam_rgb.setResolution(

#             dai.ColorCameraProperties.SensorResolution.THE_1080_P)

#         cam_rgb.setPreviewSize(HeightCamera.WIDTH, HeightCamera.HEIGHT)

 

#         try:

#             calibration = self.device.readCalibration2()

#             lens_pos = calibration.getLensPosition(dai.CameraBoardSocket.RGB)

#             if lens_pos:

#                 cam_rgb.initialControl.setManualFocus(lens_pos)

#         except Exception as e:

#             raise e

 

#         # Mono cameras

#         cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

#         cam_left.setFps(fps)

#         cam_left.setResolution(mono_resolution)

#         cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

#         cam_right.setFps(fps)

#         cam_right.setResolution(mono_resolution)

 

#         # Configure stereo node

#         node_stereo.setDefaultProfilePreset(

#             dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

#         node_stereo.setLeftRightCheck(True)

#         node_stereo.setSubpixel(True)

 

#         node_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

#         node_stereo.setOutputSize(HeightCamera.WIDTH, HeightCamera.HEIGHT)

 

#         # Configure output nodes

#         out_rgb.setStreamName("rgb")

#         out_depth.setStreamName("depth")

 

#         # Link nodes together

#         cam_rgb.preview.link(out_rgb.input)

#         cam_left.out.link(node_stereo.left)

#         cam_right.out.link(node_stereo.right)

#         node_stereo.depth.link(out_depth.input)

 

#         # Connect and start

#         self.device.startPipeline(self.pipeline)

 

#         # Get data queues

#         self.queue_depth = self.device.getOutputQueue(

#             name="depth",

#             maxSize=4,

#             blocking=False)

 

#         self.queue_rgb = self.device.getOutputQueue(

#             name="rgb",

#             maxSize=4,

#             blocking=False)

 

#     def floor_distance(self) -> float:

#         """

#         Get the distance to the floor, in millimetres.

#         """

#         if self.fixed_floor_distance != None:

#             return self.fixed_floor_distance

#         raise "Must provide a fixed distance to the floor at this time."

 

#     def mean_plateau_distance(self, depth=None) -> float:

#         """

#         Identifies a large-ish plateau in the frame (ie a region raised above the floor) and returns its mean distance from the camera in mm.

#         Optionally, you can pass a depth map to use instead of automatically calculating a fresh one.

#         """

#         if type(depth) is not np.ndarray:

#             depth = self.compute_depth_map()

 

#         # The depth map is pre-zeroed, so we're just raising/clipping everything else to 1

#         # The detection mask is 0 where there is no object, and 1 where there is

#         _, detection_mask = cv.threshold(depth, 0.1, 1, cv.THRESH_BINARY)

 

#         # Zero all the parts of the bias mask which aren't also detected depth values

#         bias_mask = cam.bias * detection_mask

 

#         # Multiply all depths by the bias, where there is data

#         masked_biased_depth = bias_mask * depth

 

#         # Get the total, biased depth of the scene

#         total_biased_depth = np.sum(masked_biased_depth)

 

#         # Convert it back to an average by dividing by the sum of however much bias mask was used

#         # In other words, the bias is a weighted sum of the depth values, but each frame there is a different number of values,

#         # so the bias map needs to be summed on every calculation

#         bias_sum = np.sum(bias_mask)

#         biased_mean_depth = total_biased_depth / bias_sum

#         return biased_mean_depth

 

#     def minimal_plateau_distance(self, depth=None) -> float:

#         """

#         Identifies a large-ish plateau in the frame (ie a region raised above the floor) and returns its nearest distance from the camera in mm, attempting to remove only obvious outliers.

#         Optionally, you can pass a depth map to use instead of automatically calculating a fresh one.

#         Returns the floor distance when nothing is detected.

#         """

#         if type(depth) is not np.ndarray:

#             depth = self.compute_depth_map()

 

#         focused_depth = depth * self.clipped_bias

#         nonzero = np.nonzero(focused_depth)

#         if len(nonzero[0]) == 0:

#             return self.floor_distance()

#         closest_depth = np.min(focused_depth[nonzero])

#         return closest_depth

 

#     def compute_depth_map(self) -> np.ndarray:

#         """

#         Get a cleaned depth map that contains the distance to raised objects, and little noise.

#         """

 

#         # For now, on every call, just get all-new data

#         if self.manage_own_capture:

#             self.flush_buffer()

 

#         # Lets always use the set amount of data

#         self.ensure_buffer_full()

 

#         depth_map = np.nanmedian(self.median_filter_buffer, axis=0)

#         depth_map[depth_map >= (

#             self.floor_distance() - self.object_noise_threshold)] = 0

 

#         return depth_map

 

#     def capture_depth_frame(self) -> np.ndarray:

#         """

#         Capture a frame of depth data. Calculating a decent depth map requires capturing multiple frames. WIll return the frame, but it may be very noisy alone.

#         """

#         frame = self.queue_depth.get().getFrame()

#         self.median_filter_buffer.append(frame)

#         while len(self.median_filter_buffer) > self.median_filter_frames:

#             self.median_filter_buffer.pop(0)

#         return frame

 

#     def capture_image_frame(self) -> np.ndarray:

#         return self.queue_rgb.get().getCvFrame()

 

#     def flush_buffer(self) -> None:

#         """

#         Empty the depth buffer of all data. Useful if the camera is known to have just received invalid data. Unaffected by whether this camera manages its own framebuffer.

#         """

#         self.median_filter_buffer = []

 

#     def ensure_buffer_full(self) -> None:

#         """

#         Fill the frame buffer with data, whether or not this camera normally manages its own frame buffer. If the buffer is already full, does *not* push any new data into it.

#         """

#         while len(self.median_filter_buffer) < self.median_filter_frames:

#             self.capture_depth_frame()

 

#     def close(self):

#         """

#         Closes the camera device.

#         """

#         self.device.close()

 

# cam = HeightCamera(manage_own_capture=False,

#                    fixed_floor_distance=750, spotlight_radius=75)

# while True:

#     cam.capture_depth_frame()

#     img = cam.capture_image_frame()

#     depth = cam.compute_depth_map()

 

#     print("------------")

#     print(

#         f"Center: {int(depth[HeightCamera.HEIGHT // 2, HeightCamera.WIDTH // 2]):4}")

#     print(f"Mean:   {int(cam.mean_plateau_distance(depth=depth)):4}")

#     print(f"Min:    {int(cam.minimal_plateau_distance(depth=depth)):4}")

 

#     cv.imshow("RGB Feed (Unaligned)", img)

#     cv.imshow("Depth Map", depth / np.max(depth))

#     cv.imshow("Focus for minimal distance picking", cam.clipped_bias)

#     cv.imshow("Bias for mean distance check", cam.bias)

 

#     if cv.waitKey(17) == ord("q"):

#         break

 

# cam.close()