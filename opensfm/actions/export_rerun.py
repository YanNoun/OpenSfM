# pyre-strict
import logging
from typing import List, Optional

import numpy as np
from opensfm import multiview, pymap, types
from opensfm.dataset import DataSet
from opensfm import pygeometry

try:
    import rerun as rr
except ImportError:
    raise ImportError(
        "Rerun is not installed. Install it with: pip install rerun-sdk"
    )

logger: logging.Logger = logging.getLogger(__name__)

# Visualization constants
GPS_COLOR = [0, 0, 255]
GCP_OBSERVATION_COLOR = [255, 0, 0]
GCP_COMPUTED_COLOR = [0, 255, 0]
GCP_REFERENCE_COLOR = [255, 255, 255]
GCP_ERROR_COLOR = [255, 0, 0]
FEATURE_COLOR = [0, 255, 0]
LABEL_COLOR = [255, 255, 255]

SIZE_TIE_POINT = 0.1

SIZE_GPS_ARROW = 1.0
SIZE_GPS_RESIDUAL_LINE = 0.05

SIZE_GCP_TARGET = 2.0
SIZE_GCP_THICKNESS = 0.05
SIZE_GCP_COMPUTED_ARROW = 0.1
SIZE_GCP_RESIDUAL_LINE = 0.02

SIZE_2D_POINT_GCP = 5.0
SIZE_2D_POINT_FEATURE = 2.0

SIZE_LABEL_SHIFT_SHOT = 0.5

IMAGE_PLANE_DISTANCE = 2.0
MAX_IMAGE_WIDTH = 1500


def run_dataset(
    data: DataSet,
    output: Optional[str] = None,
    reconstruction_index: int = 0,
) -> None:
    """Export reconstruction to Rerun format for 3D visualization.

    Args:
        data: OpenSfM dataset
        output: Output .rrd file path (default: dataset_path/rerun.rrd)
        reconstruction_index: Index of reconstruction to export (default: 0)
    """

    # Load reconstruction
    reconstructions = data.load_reconstruction()
    if not reconstructions:
        logger.error("No reconstructions found in dataset")
        return

    if reconstruction_index >= len(reconstructions):
        logger.error(
            f"Reconstruction index {reconstruction_index} out of range "
            f"(found {len(reconstructions)} reconstructions)"
        )
        return

    reconstruction = reconstructions[reconstruction_index]
    logger.info(
        f"Exporting reconstruction {reconstruction_index} with "
        f"{len(reconstruction.shots)} shots, "
        f"{len(reconstruction.points)} points"
    )

    # Initialize Rerun
    output_path = output or data.data_path + "/rerun.rrd"
    rr.init("OpenSfM Reconstruction", spawn=False)
    rr.save(output_path)
    logger.info(f"Saving Rerun data to {output_path}")

    # Set up coordinate system (OpenSfM uses East-North-Up)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # Center the view on the median of the reconstruction points
    if reconstruction.points:
        points = np.array([p.coordinates for p in reconstruction.points.values()])
        median_center = np.median(points, axis=0)
        rr.log("world", rr.Transform3D(translation=-median_center))

    # Export reference coordinate system
    if reconstruction.reference:
        ref = reconstruction.reference
        rr.log(
            "world/reference",
            rr.TextDocument(
                f"Reference LLA:\n"
                f"Latitude: {ref.lat:.6f}°\n"
                f"Longitude: {ref.lon:.6f}°\n"
                f"Altitude: {ref.alt:.2f}m",
            ),
        )

    # Export cameras (intrinsics)
    _export_cameras(reconstruction)

    # Export camera poses (shots)
    _export_shots(data, reconstruction)

    # Export 3D points
    _export_points(reconstruction)

    # Export Ground Control Points
    gcp = data.load_ground_control_points()
    if gcp:
        _export_gcp(data, reconstruction, gcp)

    # Export tracks manager (optional, for visualizing observations)
    # if data.tracks_exists():
    #     tracks_manager = data.load_tracks_manager()
    #     _export_observations(reconstruction, tracks_manager)

    logger.info(f"Rerun export completed: {output_path}")
    logger.info(f"Open with: rerun {output_path}")


def _get_shot_sequence_id(shot_id: str) -> int:
    """Extract integer ID from shot_id string (e.g., 'Shot 123' -> 123).
    
    If extraction fails, use hash of the shot_id to ensure uniqueness.
    """
    try:
        # Try to extract number from strings like "Shot 123" or "DSC_0123.jpg"
        import re
        numbers = re.findall(r'\d+', shot_id)
        if numbers:
            return int(numbers[-1])  # Use the last number found
    except:
        pass
    
    # Fallback: use hash of shot_id
    return hash(shot_id) & 0x7FFFFFFF  # Ensure positive integer


def _get_scaled_dimensions(width: int, height: int) -> tuple[int, int]:
    if width > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / width
        return int(width * scale), int(height * scale)
    return width, height


def _get_camera_calibration(camera: pygeometry.Camera, width: int, height: int):
    """Calculate camera intrinsic parameters."""
    fx = fy = camera.focal * max(width, height)
    if hasattr(camera, "focal_x") and hasattr(camera, "focal_y"):
        fx = camera.focal_x * max(width, height)
        fy = camera.focal_y * max(width, height)

    cx = width / 2.0
    cy = height / 2.0
    if hasattr(camera, "c_x") and hasattr(camera, "c_y"):
        cx = camera.c_x * max(width, height) + width / 2.0
        cy = camera.c_y * max(width, height) + height / 2.0
    return fx, fy, cx, cy


def _export_cameras(reconstruction: types.Reconstruction) -> None:
    """Export camera models (intrinsics)."""
    for camera_id, camera in reconstruction.cameras.items():
        # Get camera parameters
        width = int(camera.width)
        height = int(camera.height)
        width, height = _get_scaled_dimensions(width, height)
        fx, fy, cx, cy = _get_camera_calibration(camera, width, height)
        focal_length_px = camera.focal * max(width, height)

        # Log camera intrinsics (strip spaces from camera_id)
        camera_id_clean = camera_id.replace(" ", "_")
        rr.log(
            f"world/cameras/{camera_id_clean}/info",
            rr.TextDocument(
                f"Camera: {camera_id}\n"
                f"Type: {camera.projection_type}\n"
                f"Resolution: {width}x{height}\n"
                f"Focal: {focal_length_px:.1f}px\n"
                f"Principal Point: ({cx:.1f}, {cy:.1f})"
            ),
        )


def _export_shots(data: DataSet, reconstruction: types.Reconstruction) -> None:
    """Export camera shots (poses and images)."""
    reference = reconstruction.reference

    for shot_id, shot in reconstruction.shots.items():
        _export_shot_pose(shot, shot_id)

        # Collect GPS position and error line if available
        if reference and shot.metadata and shot.metadata.gps_position.has_value:
            gps_topo = shot.metadata.gps_position.value
            _export_shot_gps(shot, shot_id, gps_topo)

        _export_shot_pinhole(data, shot, shot_id)


def _export_shot_pose(shot: pymap.Shot, shot_id: str) -> None:
    pose = shot.pose
    R = pose.get_rotation_matrix()  # 3x3 rotation matrix
    t = pose.get_origin()  # 3D position (camera center in world coords)

    # Rerun expects rotation from world to camera
    # OpenSfM GetRotationMatrix gives camera-to-world rotation
    R_world_from_camera = R.T  # Transpose to get world-from-camera

    # Strip spaces from camera_id for clean entity paths
    camera_id_clean = shot.camera.id.replace(" ", "_")

    # Log camera pose with integer sequence ID
    sequence_id = _get_shot_sequence_id(shot_id)
    rr.set_time_sequence("shot", sequence_id)
    rr.log(
        f"world/cameras/{camera_id_clean}/shots/{shot_id}/pose",
        rr.Transform3D(
            translation=t,
            mat3x3=R_world_from_camera,
            from_parent=False,
        ),
    )

    labels_shift = np.array([0, 0, SIZE_LABEL_SHIFT_SHOT])  # Shift labels above targets
    rr.log(
        f"world/cameras/{camera_id_clean}/shots/{shot_id}/pose",
        rr.Points3D(
            positions=labels_shift,
            labels=[shot_id],
            radii=0.0,  # Invisible points, just for labels
            colors=[LABEL_COLOR],  # White labels
        ),
        static=True,
    )


def _export_shot_pinhole(data: DataSet, shot: pymap.Shot, shot_id: str) -> None:
    # Create pinhole camera for image projection
    width = int(shot.camera.width)
    height = int(shot.camera.height)
    width, height = _get_scaled_dimensions(width, height)
    fx, fy, cx, cy = _get_camera_calibration(shot.camera, width, height)

    camera_id_clean = shot.camera.id.replace(" ", "_")
    rr.log(
        f"world/cameras/{camera_id_clean}/shots/{shot_id}/pose/pinhole",
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            image_plane_distance=IMAGE_PLANE_DISTANCE,
        ),
    )

    # Load and log image if available
    try:
        image = data.load_image(shot_id)
        
        # Resize image if needed
        if image.shape[1] != width or image.shape[0] != height:
            import cv2
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        rr.log(
            f"world/cameras/{camera_id_clean}/shots/{shot_id}/pose/pinhole/image",
            rr.Image(image).compress(jpeg_quality=50),
        )
    except Exception as e:
        logger.warning(f"Could not load image for {shot_id}: {e}")


def _export_shot_gps(
    shot: pymap.Shot,
    shot_id: str,
    gps_topo: np.ndarray,
) -> None:
    camera_id_clean = shot.camera.id.replace(" ", "_")
    base_path = f"world/cameras/{camera_id_clean}/shots/{shot_id}/gps"

    arrow_size = SIZE_GPS_ARROW
    arrow_radii = arrow_size / 8.0
    arrow_length = arrow_size  # Length of arrows for GPS positions
    
    origin = gps_topo.copy()
    origin[2] += arrow_length
    vector = np.array([0, 0, -arrow_length])

    rr.log(
        f"{base_path}/position",
        rr.Arrows3D(
            origins=[origin],
            vectors=[vector],
            colors=[GPS_COLOR],  # Blue
            radii=arrow_radii,
        ),
        static=True,
    )

    rr.log(
        f"{base_path}/residual",
        rr.LineStrips3D(
            [[shot.pose.get_origin(), gps_topo]],
            colors=[GPS_COLOR],  # Blue
            radii=SIZE_GPS_RESIDUAL_LINE,
        ),
        static=True,
    )


def _export_points(reconstruction: types.Reconstruction) -> None:
    """Export 3D points (structure)."""

    if not reconstruction.points:
        return

    # Collect all points
    positions = []
    colors = []
    point_ids = []

    for point_id, point in reconstruction.points.items():
        positions.append(point.coordinates)
        # Convert color from 0-1 to 0-255 if needed
        color = point.color
        if all(c <= 1.0 for c in color):
            color = [int(c * 255) for c in color]
        colors.append(color)
        point_ids.append(point_id)

    # Log all points at once for efficiency
    rr.log(
        "world/points/automatic",
        rr.Points3D(
            positions=np.array(positions),
            colors=np.array(colors, dtype=np.uint8),
            radii=SIZE_TIE_POINT,  # Small radius for tie points
        ),
        static=True,
    )

    logger.info(f"Exported {len(positions)} automatic tie points")


def _export_gcp(
    data: DataSet,
    reconstruction: types.Reconstruction,
    gcp: List[pymap.GroundControlPoint],
) -> None:
    """Export Ground Control Points."""
    if not gcp:
        return

    reference = reconstruction.reference

    for point in gcp:
        # Get 3D position
        if point.lla:
            # Convert from lat/lon/alt to topocentric coordinates
            lla_vec = point.lla_vec
            pos = reference.to_topocentric(*lla_vec)

            # Try to triangulate GCP
            triangulated = multiview.triangulate_gcp(
                point, reconstruction.shots, data.config["gcp_reprojection_error_threshold"]
            )

            _export_single_gcp_geometry(point.id, pos, triangulated)
            _export_gcp_observations(reconstruction, point)

    logger.info(f"Exported {len(gcp)} Ground Control Points")


def _export_gcp_observations(
    reconstruction: types.Reconstruction, 
    point: pymap.GroundControlPoint
) -> None:
    # Log GCP observations in images
    for obs in point.observations:
        shot_id = obs.shot_id
        if shot_id in reconstruction.shots:
            # Log 2D observation
            shot = reconstruction.shots[shot_id]
            width = int(shot.camera.width)
            height = int(shot.camera.height)
            width, height = _get_scaled_dimensions(width, height)
            normalize = max(width, height)

            # Convert from normalized to pixel coordinates
            px = obs.projection[0] * normalize + width / 2.0
            py = obs.projection[1] * normalize + height / 2.0

            # Strip spaces from camera_id
            camera_id_clean = shot.camera.id.replace(" ", "_")

            # Set time sequence with integer ID
            sequence_id = _get_shot_sequence_id(shot_id)
            rr.set_time_sequence("shot", sequence_id)
            rr.log(
                f"world/cameras/{camera_id_clean}/shots/{shot_id}/pinhole/gcp/{point.id}",
                rr.Points2D(
                    [[px, py]],
                    colors=[GCP_OBSERVATION_COLOR],
                    radii=SIZE_2D_POINT_GCP,
                    labels=[point.id],
                ),
            )


def _export_single_gcp_geometry(
    gcp_id: str,
    reference_pos: np.ndarray,
    computed_pos: Optional[np.ndarray],
) -> None:
    base_path = f"world/gcp/{gcp_id}"

    # Target dimensions (1m x 1m target)
    target_size = SIZE_GCP_TARGET
    quad_r = target_size / 4.0  # Half-size of a quadrant
    thickness = SIZE_GCP_THICKNESS

    pos_np = np.array(reference_pos)
    color = GCP_REFERENCE_COLOR if computed_pos is not None else GCP_ERROR_COLOR

    # Checkerboard boxes
    centers = [
        pos_np + [-quad_r, quad_r, 0],
        pos_np + [quad_r, -quad_r, 0]
    ]
    half_sizes = [[quad_r, quad_r, thickness]] * 2
    colors = [color] * 2

    rr.log(
        f"{base_path}/marker",
        rr.Boxes3D(
            centers=centers,
            half_sizes=half_sizes,
            colors=colors,
            fill_mode="solid",
        ),
        static=True,
    )

    # Outline
    rr.log(
        f"{base_path}/outline",
        rr.Boxes3D(
            centers=[pos_np],
            half_sizes=[[target_size / 2.0, target_size / 2.0, thickness]],
            colors=[color],
            radii=thickness,
            fill_mode="major_wireframe",
        ),
        static=True,
    )

    # Label
    labels_shift = np.array([0, 0, target_size])
    rr.log(
        f"{base_path}/label",
        rr.Points3D(
            positions=[pos_np + labels_shift],
            labels=[gcp_id],
            radii=0.0,
            colors=[LABEL_COLOR],
        ),
        static=True,
    )

    # Computed position
    if computed_pos is not None:
        arrow_length = 0.5
        origin = computed_pos.copy()
        origin[2] += arrow_length
        vector = np.array([0, 0, -arrow_length])

        rr.log(
            f"{base_path}/computed",
            rr.Arrows3D(
                origins=[origin],
                vectors=[vector],
                colors=[GCP_COMPUTED_COLOR],
                radii=SIZE_GCP_COMPUTED_ARROW,
            ),
            static=True,
        )

        # Residual line
        rr.log(
            f"{base_path}/residual",
            rr.LineStrips3D(
                [[computed_pos, reference_pos]],
                colors=[GCP_COMPUTED_COLOR],
                radii=SIZE_GCP_RESIDUAL_LINE,
            ),
            static=True,
        )


def _export_observations(
    reconstruction: types.Reconstruction,
    tracks_manager: pymap.TracksManager,
) -> None:
    """Export feature observations (2D points in images)."""

    logger.info("Exporting feature observations...")

    # Sample some tracks to avoid overwhelming the viewer
    max_tracks = 1000
    track_ids = list(reconstruction.points.keys())
    if len(track_ids) > max_tracks:
        import random

        track_ids = random.sample(track_ids, max_tracks)

    for track_id in track_ids:
        if track_id not in reconstruction.points:
            continue

        observations = tracks_manager.get_track_observations(track_id)

        for shot_id, obs in observations.items():
            if shot_id not in reconstruction.shots:
                continue

            shot = reconstruction.shots[shot_id]
            width = int(shot.camera.width)
            height = int(shot.camera.height)
            width, height = _get_scaled_dimensions(width, height)
            normalize = max(width, height)

            # Convert from normalized to pixel coordinates
            px = obs.point[0] * normalize + width / 2.0
            py = obs.point[1] * normalize + height / 2.0

            # Strip spaces from camera_id
            camera_id_clean = shot.camera.id.replace(" ", "_")

            # Set time sequence with integer ID
            sequence_id = _get_shot_sequence_id(shot_id)
            rr.set_time_sequence("shot", sequence_id)
            rr.log(
                f"world/cameras/{camera_id_clean}/shots/{shot_id}/pinhole/features",
                rr.Points2D(
                    [[px, py]],
                    colors=[FEATURE_COLOR],  # Green for features
                    radii=SIZE_2D_POINT_FEATURE,
                ),
            )

    logger.info(f"Exported observations for {len(track_ids)} tracks")