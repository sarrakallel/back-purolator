from typing import List, Tuple
import numpy as np
import depthai as dai
import cv2 as cv
from math import pi, cos, sin
from rest_framework import serializers


class MeasurementCamera:

    """
    This class manages an OAK camera, and computes depth information from its input. It assuems it is looking down at an object on a flat floor, and can compute distances to the object.

    To compute depth data, the class needs to record multiple frames of input and filter them. It can either manage its own capture intervals or let the caller do it. When managing its own capture intervals, depth queries may take longer, as it needs time to capture data. If this is an issue, then set manage_own_capture to False, and periodically call capture_depth_frame.
    """

    # Using 640x400 because its an integer ratio of the camera capture size
    # Non-integer ratios get weird with the matrix and centre scaling, so we simply will not
    WIDTH = 640
    HEIGHT = 400

    def __init__(self,
                 manage_own_capture=True,
                 median_filter_frames=5,
                 fixed_floor_distance=None,
                 object_noise_threshold=50,
                 spotlight_radius=75,
                 fps=15):
        """
        manage_own_capture: When True, the class will record frames of data whenever distance methods are called. When False, the caller must make calls to capture_depth_frame() periodically.
        median_filter_frames: How long an interval the algorithm should consider when computing depths.
        fixed_floor_distance: Set this to hardcode a distance to the floor in mm.
        object_noise_threshold: Minimum height an object detection must be to register in mm. Needs to be larger for higher mount points of the camera.
        spotlight_radius: configures the bias region, everything inside this radius from the centre of the image will be considered for minimal distance calculations, and that radius is most strongly considered for mean distance calculations
        """
        self.manage_own_capture = manage_own_capture
        self.median_filter_frames = median_filter_frames
        self.median_filter_buffer = []
        self.fixed_floor_distance = fixed_floor_distance
        self.object_noise_threshold = object_noise_threshold

        mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

        # Bias function for sampling, does not guarantee all the values sum to 1
        # (would be a pointless optimization since different images will mask out different parts of the function)
        # Use an exponential falloff about the centre

        cx = MeasurementCamera.WIDTH // 2
        cy = MeasurementCamera.HEIGHT // 2
        # the alpha value stretches out the region of bias
        # alpha = radius at which pixel contribution is halved from centre of image
        alpha = 1.0 / spotlight_radius
        self.bias = np.array(
            [np.array(
                [(2 / (1 + alpha**2*((x-cx)**2 + (y-cy)**2)))  # 2.0 at centre, 1.0 at r=alpha, decays outward
                 for x in range(MeasurementCamera.WIDTH)])
             for y in range(MeasurementCamera.HEIGHT)])

        # Make a spotlight for minimum distance checks
        _, self.clipped_bias = cv.threshold(
            self.bias, 1.0, 1.0, cv.THRESH_BINARY)

        # DepthAI manager classes
        self.pipeline = dai.Pipeline()
        self.device = dai.Device()

        # Pipeline nodes
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_left = self.pipeline.create(dai.node.MonoCamera)
        cam_right = self.pipeline.create(dai.node.MonoCamera)
        node_stereo = self.pipeline.create(dai.node.StereoDepth)
        out_rgb = self.pipeline.create(dai.node.XLinkOut)
        out_depth = self.pipeline.create(dai.node.XLinkOut)
        out_right = self.pipeline.create(dai.node.XLinkOut)

        # Configure cameras
        # RGB
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setFps(fps)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(MeasurementCamera.WIDTH,
                               MeasurementCamera.HEIGHT)

        try:
            calibration = self.device.readCalibration2()
            lens_pos = calibration.getLensPosition(dai.CameraBoardSocket.RGB)
            if lens_pos:
                cam_rgb.initialControl.setManualFocus(lens_pos)
        except Exception as e:
            raise e

        # Mono cameras
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setFps(fps)
        cam_left.setResolution(mono_resolution)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setFps(fps)
        cam_right.setResolution(mono_resolution)

        # Configure stereo node
        node_stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        node_stereo.setLeftRightCheck(True)
        node_stereo.setSubpixel(True)

        node_stereo.setDepthAlign(
            dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT)
        node_stereo.setOutputSize(
            MeasurementCamera.WIDTH, MeasurementCamera.HEIGHT)
        node_stereo.setOutputKeepAspectRatio(True)
        node_stereo.setRectification(True)

        # Configure output nodes
        out_rgb.setStreamName("rgb")
        out_depth.setStreamName("depth")
        out_right.setStreamName("right")

        # Link nodes together
        cam_rgb.preview.link(out_rgb.input)
        cam_left.out.link(node_stereo.left)
        cam_right.out.link(out_right.input)
        cam_right.out.link(node_stereo.right)
        node_stereo.depth.link(out_depth.input)

        # Connect and start
        self.device.startPipeline(self.pipeline)

        # Get data queues
        self.queue_depth = self.device.getOutputQueue(
            name="depth",
            maxSize=4,
            blocking=False)

        self.queue_rgb = self.device.getOutputQueue(
            name="rgb",
            maxSize=4,
            blocking=False)

        self.queue_right = self.device.getOutputQueue(
            name="right",
            maxSize=4,
            blocking=False)

        # WOrk out camera parameters for putting image points into camera space
        calibration = self.device.readCalibration()
        intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT)

        # Instrinsic matrix is for 1280x800 camera
        # Adjust by ratio of resolutions: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        fx = MeasurementCamera.WIDTH / 1280
        fy = MeasurementCamera.HEIGHT / 800

        # Actually adjust the intrinsic matrix:
        intrinsics[0][0] *= fx
        intrinsics[0][2] *= fx
        intrinsics[1][1] *= fy
        intrinsics[1][2] *= fy

        # Good enough for now, still need undistort and pose matrix
        self.m_projection = np.array(intrinsics)
        self.m_unprojection = np.linalg.inv(self.m_projection)

        # Each pixel in the image corresponds to a ray stretching out into camera space
        # Make a map of this. the map can be element-wise multiplied by a depth map to get a 2d x,y,z point cloud, which can be flattened for pure points as needed
        self.camera_ray_map = np.array([
            [self.m_unprojection @ np.array([x, y, 1])
             for x in range(MeasurementCamera.WIDTH)]
            for y in range(MeasurementCamera.HEIGHT)])

        # Pallet measurements
        self.pallet_angle = None
        self.pallet_x = None
        self.pallet_y = None

    def px_to_pos(self, x: int, y: int, depth: float) -> np.array:
        """
        Takes an x/y pixel and its depth, converts to an x,y,z vector in camera space.
        """
        return self.camera_ray_map[y][x] * depth

    def floor_distance(self) -> float:
        """
        Get the distance to the floor, in millimetres.
        """
        if self.fixed_floor_distance != None:
            return self.fixed_floor_distance
        raise "Must provide a fixed distance to the floor at this time."

    def compute_detection_map(self, depth=None) -> Tuple[np.ndarray, dict]:
        """
        Finds the raised part of the image likely to be of interest, and masks it to a 0/1 image. Also returns a dictionary describing the rectangular region, in pixel coordinates.
        """
        if Images.is_none(depth):
            depth = self.compute_depth_map()

        # The depth map is pre-zeroed for out of bounds data, so we're just raising/clipping everything else to 1
        # The detection mask is 0 where there is no object, and 1 where there is
        _, detection_mask = cv.threshold(depth, 0.1, 1, cv.THRESH_BINARY)
        detection_mask = detection_mask.astype("uint8")

        # Despeckle noise out
        # ie, erode and then dialate (== "opening"), does not impact overall size
        # because the dialate step undoes anything chipped away by erode
        # (slightly smooths corners but not important)
        radius = 10
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
        despeckle_mask = cv.morphologyEx(detection_mask, cv.MORPH_OPEN, kernel)
        detection_mask = cv.bitwise_and(
            detection_mask, detection_mask, mask=despeckle_mask)

        # Look for blob in middle
        contours, _ = cv.findContours(
            detection_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

        # Find the largest likely candidate contour based on size and distance from center
        # Will pick the largest matching one
        min_area_px = 5000
        max_distance_center = MeasurementCamera.WIDTH * 0.25

        biggest_rect = None
        biggest_rect_area = 0
        for contour in contours:
            rect = cv.minAreaRect(contour)
            center, size, _ = rect

            # The detected box must be near the center of the screen and fairly large
            area = size[0] * size[1]
            if area < min_area_px:
                continue
            distance = ((center[0] - MeasurementCamera.WIDTH / 2) **
                        2 + (center[1] - MeasurementCamera.HEIGHT / 2)**2) ** 0.5
            if distance > max_distance_center:
                continue

            if area > biggest_rect_area:
                biggest_rect_area = area
                biggest_rect = rect

        if biggest_rect == None:
            return None, None

        # Create a mask containing just that one contour's rotated bounding box
        box = np.intp(cv.boxPoints(biggest_rect))
        detection_mask = np.zeros_like(detection_mask)
        cv.drawContours(detection_mask, [box], -1, 1, thickness=cv.FILLED)

        center, size, angle = biggest_rect
        detection_characteristics = {
            "center": center,
            "size": size,
            "angle": angle,
            "corners": box
        }
        return detection_mask, detection_characteristics

    def measure_pallet(self, depth=None) -> List:
        """
        Assuming the pallet is in view, get its u/v dimensions. Also stores the angle and centre for later load measurements.
        """
        if Images.is_none(depth):
            depth = self.compute_depth_map()

        mask, characteristics = self.compute_detection_map(depth=depth)

        # Get the z-value of the mask, assume it is constant
        # We do this averaging because there can be noise right around the corners
        # of the detected rectangle mask, so its simpler to assume the whole pallet
        # is flat (quite reasonably, really)
        distance = self.mean_nonzero_value(depth, mask)

        # Use it to find where in three-space all the corners are
        points = [self.camera_ray_map[y][x] *
                  distance for (x, y) in characteristics["corners"]]

        # Will get four points, for six unique distances
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        lengths = sorted([np.linalg.norm(points[a] - points[b])
                         for (a, b) in edges])

        dims = [lengths[0], lengths[2]]
        angle = characteristics["angle"]
        x = characteristics["center"][0]
        y = characteristics["center"][1]

        self.pallet_angle = angle
        self.pallet_x = x
        self.pallet_y = y
        # print(
        #     f"{int(dims[0])/10} cm x {int(dims[1])/10} cm @ ({x:04f}, {y:04f}) px {int(angle*10)/10} deg")
        return dims

    def has_measured_pallet(self) -> bool:
        """
        Checks to see if the pallet has been measured at least once.
        """
        return self.pallet_angle != None

    def clear_pallet_measurement(self):
        """
        Deletes the current pallet measurement
        """
        self.pallet_angle = None
        self.pallet_x = None
        self.pallet_y = None

    def measure_load(self, depth=None):
        """
        Looks for the loaded pallet in the depth map, and measures it in the previously measured pallet-space (ie bounds in terms of the pallet's width and length directions). Does not save the values, but returns min_u, max_u, min_v, max_v

        Returns the min and max instead of the difference in case the caller wants to compute bounds over many measurements rather than just the one. If only interested in the currently visible portion, then the "width" and "height" are as simple as max_u - min_u and max_v - min_v.

        Note that the "angle" of the pallet is not very consistent. This is okay since we only need to measure it once, but means that the u and v directions may not be consistent between pallets.
        """

        if not self.has_measured_pallet():
            raise "Must call measure_pallet() on the empty pallet before attempting to measure the load"

        angle = self.pallet_angle
        center_loc = [self.pallet_x, self.pallet_y]

        if Images.is_none(depth):
            depth = self.compute_depth_map()

        # Get the big blob of depth in the middle of the frame, this will be the pallet (whether it is loaded or unloaded)
        mask, characteristics = self.compute_detection_map(depth=depth)

        # Get all the depth values likely on the pallet
        load_depth = depth * mask
        # Make a point cloud (a 1d array of vectors)
        load_points = (self.camera_ray_map *
                       load_depth[..., np.newaxis]).reshape(load_depth.shape[0] * load_depth.shape[1], 3)

        # Remove all the zero vectors, since they were outside the mask
        load_points = load_points[~np.all(np.isclose(load_points, 0), axis=1)]

        # Rotate all the points about the centre by the given angle
        # Start by translating to origin
        load_points -= np.array([center_loc[0], center_loc[1], 0])
        # Then rotate about z axis
        theta = (-angle) * pi / 180
        m = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta),  cos(theta), 0],
            [0,                    0, 1]])
        # Do the transform
        load_points = np.matmul(m, load_points.T).T

        # Then translate back
        load_points += np.array([center_loc[0], center_loc[1], 0])

        us = load_points[:, 0]
        vs = load_points[:, 1]
        # zs = load_points[:, 2]

        min_u = np.min(us)
        max_u = np.max(us)

        min_v = np.min(vs)
        max_v = np.max(vs)

        # min_z = np.min(zs)
        # max_z = np.max(zs)

        # print(f"{int(min_x)} -> {int(max_x)} = {int(max_x - min_x)} mm")
        # print(f"{int(min_y)} -> {int(max_y)} = {int(max_y - min_y)} mm")
        # print(f"{int(min_z)} -> {int(max_z)} = {int(max_z - min_z)} mm")

        return min_u, max_u, min_v, max_v

    def mean_plateau_distance(self, depth=None) -> float:
        """
        Identifies a large-ish plateau in the frame (ie a region raised above the floor) and returns its mean distance from the camera in mm.
        Optionally, you can pass a depth map to use instead of automatically calculating a fresh one.
        """
        if Images.is_none(depth):
            depth = self.compute_depth_map()

        # The depth map is pre-zeroed, so we're just raising/clipping everything else to 1
        # The detection mask is 0 where there is no object, and 1 where there is
        _, detection_mask = cv.threshold(depth, 0.1, 1, cv.THRESH_BINARY)

        # Zero all the parts of the bias mask which aren't also detected depth values
        bias_mask = cam.bias * detection_mask

        # Multiply all depths by the bias, where there is data
        masked_biased_depth = bias_mask * depth

        # Get the total, biased depth of the scene
        total_biased_depth = np.sum(masked_biased_depth)

        # Convert it back to an average by dividing by the sum of however much bias mask was used
        # In other words, the bias is a weighted sum of the depth values, but each frame there is a different number of values,
        # so the bias map needs to be summed on every calculation
        bias_sum = np.sum(bias_mask)
        biased_mean_depth = total_biased_depth / bias_sum
        return biased_mean_depth

    def minimal_plateau_distance(self, depth=None) -> float:
        """
        Identifies a large-ish plateau in the frame (ie a region raised above the floor) and returns its nearest distance from the camera in mm, attempting to remove only obvious outliers.
        Optionally, you can pass a depth map to use instead of automatically calculating a fresh one.
        Returns the floor distance when nothing is detected.
        """
        if Images.is_none(depth):
            depth = self.compute_depth_map()

        focused_depth = depth * self.clipped_bias
        nonzero = np.nonzero(focused_depth)
        if len(nonzero[0]) == 0:
            return self.floor_distance()
        closest_depth = np.min(focused_depth[nonzero])
        return closest_depth

    def compute_depth_map(self) -> np.ndarray:
        """
        Get a cleaned depth map that contains the distance to raised objects, and little noise.
        """

        # For now, on every call, just get all-new data
        if self.manage_own_capture:
            self.flush_buffer()

        # Lets always use the set amount of data
        self.ensure_buffer_full()

        depth_map = np.nanmedian(self.median_filter_buffer, axis=0)
        depth_map[depth_map >= (
            self.floor_distance() - self.object_noise_threshold)] = 0

        return depth_map

    def mean_nonzero_value(self, img, mask=None) -> float:
        """
        Takes an image and optionally a mask, computes the average value in the (masked parts of the) image.
        """
        if Images.is_none(mask):
            mask = np.ones_like(img)

        masked = img * mask
        sum = np.sum(masked)
        nonzero_count = np.count_nonzero(masked)
        return sum / nonzero_count

    def capture_depth_frame(self) -> np.ndarray:
        """
        Capture a frame of depth data. Calculating a decent depth map requires capturing multiple frames. WIll return the frame, but it may be very noisy alone.
        """
        frame = self.queue_depth.get().getFrame()
        self.median_filter_buffer.append(frame)
        while len(self.median_filter_buffer) > self.median_filter_frames:
            self.median_filter_buffer.pop(0)
        return frame

    def capture_image_frame(self) -> np.ndarray:
        """
        Get the RGB frame.
        """
        return self.queue_rgb.get().getCvFrame()

    def capture_eye_frame(self) -> np.ndarray:
        """
        Get the grayscale eye that aligns with the depth map.
        """
        return self.queue_right.get().getCvFrame()

    def flush_buffer(self) -> None:
        """
        Empty the depth buffer of all data. Useful if the camera is known to have just received invalid data. Unaffected by whether this camera manages its own framebuffer.
        """
        self.median_filter_buffer = []

    def ensure_buffer_full(self) -> None:
        """
        Fill the frame buffer with data, whether or not this camera normally manages its own frame buffer. If the buffer is already full, does *not* push any new data into it.
        """
        while len(self.median_filter_buffer) < self.median_filter_frames:
            self.capture_depth_frame()

    def close(self):
        """
        Closes the camera device.
        """
        self.device.close()


class Images:
    def add_crosshair(img: np.ndarray, x_px, y_px, color):
        """
        Adds a + to an image, modifying the original.
        """
        x_px = int(x_px)
        y_px = int(y_px)

        img[y_px-10:y_px+10, x_px-2:x_px+2] = color
        img[y_px-2:y_px+2,  x_px-10:x_px+10] = color

    def add_text(img: np.ndarray, text, x, y, size, color):
        """
        Write onto an image. All units are in pixels (unlike the opencv interface).
        """
        cv.putText(img, text, (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, size / 30, color, 2, cv.LINE_AA)

    def normalized_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Returns a copy of the input image, normalized and made into an 8 bit grayscale.
        """
        return (255 * img / np.max(img)).astype("uint8")

    def is_some(img_or_none) -> bool:
        """
        Checks to see if a variable is an np.ndarray or None. Because numpy overrides the equality operator, the check is a bit ugly.
        """
        return type(img_or_none) == np.ndarray

    def is_none(img_or_none) -> bool:
        """
        Checks to see if a variable is an np.ndarray or None. Because numpy overrides the equality operator, the check is a bit ugly.
        """
        return type(img_or_none) == type(None)


cam = MeasurementCamera(manage_own_capture=False,
                        fixed_floor_distance=710, spotlight_radius=75)
print("Initialized Camera")

cursor_x = 320
cursor_y = 240


def on_click(ev, x, y, flags, param):
    global cursor_x, cursor_y
    if ev == cv.EVENT_LBUTTONDOWN:
        cursor_x = x
        cursor_y = y


cv.namedWindow("Depth Map")
cv.setMouseCallback("Depth Map", on_click)

while True:

    # Always show the RGB video feed
    img = cam.capture_image_frame()
    cv.imshow("RGB Feed (Unaligned)", img)

    # Calculate the depth map and look for something pallet-like in the frame
    cam.capture_depth_frame()
    depth = cam.compute_depth_map()
    detection, _ = cam.compute_detection_map(depth)

    # If we found something, then render the output
    if Images.is_some(detection):

        # Locate the cursor in 3D camera space (mm), will default to 0, 0, 0 when there is no detection
        cz = depth[cursor_y][cursor_x]
        xyz = cam.px_to_pos(cursor_x, cursor_y, cz)

        # Stack the grayscale video feed, pallet detection mask, and filtered depth map into one image
        depth_render = cv.merge([
            cam.capture_eye_frame(),
            Images.normalized_grayscale(detection),
            Images.normalized_grayscale(depth)])

        # Draw the cursor to it
        Images.add_crosshair(depth_render, cursor_x, cursor_y, [0, 0, 255])

        # Label the x/y/z of the cursor in camera space
        Images.add_text(
            depth_render, f"Cursor: {int(xyz[0]):4}, {int(xyz[1]):4}, {int(xyz[2]):4} mm", 5, 20, 20, (127, 255, 127))

        # Once the pallet has been set:
        if cam.has_measured_pallet():
            # Display the pallet data that was captured
            Images.add_text(depth_render,
                            f"Pallet: x: {int(cam.pallet_x):3} y: {int(cam.pallet_y):3} mm @ {int(cam.pallet_angle*10)/10} deg",
                            5, 40,
                            20, (127, 255, 127))
            # Display the current load bounds in the pallet's uv dimensions
            bounds = cam.measure_load(depth)
            du = bounds[1] - bounds[0]
            dv = bounds[3] - bounds[2]
            Images.add_text(depth_render,
                            f"Load: {int(du):3} x {int(dv):3} mm",
                            5, 60,
                            20, (127, 255, 127))

        # Render output
        cv.imshow("Depth Map", depth_render)

    # Press 'p' to take the current detection as the pallet baseline
    if cv.waitKey(17) == ord("p"):
        cam.measure_pallet(depth)

    # Good to quit gracefully because it closes the camera
    if cv.waitKey(17) == ord("q"):
        break


cam.close()
print("Closed camera")


# cam = MeasurementCamera(manage_own_capture=False,
#                          fixed_floor_distance=710, spotlight_radius=75)
# while True:
#     cam.capture_depth_frame()
#     img = cam.capture_image_frame()
#     depth = cam.compute_depth_map()

#     print("------------")
#     print(
#         f"Center: {int(depth[MeasurementCamera.HEIGHT // 2, MeasurementCamera.WIDTH // 2]):4}")
#     print(f"Mean:   {int(cam.mean_plateau_distance(depth=depth)):4}")
#     print(f"Min:    {int(cam.minimal_plateau_distance(depth=depth)):4}")

#     cv.imshow("RGB Feed (Unaligned)", img)
#     cv.imshow("Depth Map", depth / np.max(depth))
#     cv.imshow("Focus for minimal distance picking", cam.clipped_bias)
#     cv.imshow("Bias for mean distance check", cam.bias)

#     if cv.waitKey(17) == ord("q"):
#         break

# cam.close()



class MeasurementSerializer(serializers.Serializer):
    width = serializers.IntegerField()
    height = serializers.IntegerField()
    # Add more fields as needed for your measurements
