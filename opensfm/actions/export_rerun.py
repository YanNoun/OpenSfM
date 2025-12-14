# pyre-strict
import logging
from typing import List, Optional

import numpy as np
from opensfm import features, multiview, pymap, types
from opensfm.dataset import DataSet
from opensfm import pygeometry

import matplotlib.cm as cm
import rerun as rr
import rerun.blueprint as rrb

logger: logging.Logger = logging.getLogger(__name__)

# Visualization constants
GPS_COLOR = [100, 180, 255]
GCP_OBSERVATION_COLOR = [255, 160, 120]
GCP_COMPUTED_COLOR = [120, 255, 140]
GCP_REFERENCE_COLOR = [255, 230, 100] 
GCP_ERROR_COLOR = [255, 100, 100]
FEATURE_COLOR = [100, 255, 220] 
LABEL_COLOR = [220, 220, 220] 

SIZE_TIE_POINT = 0.2

SIZE_GPS_ARROW = 1.0
SIZE_GPS_RESIDUAL_LINE = 0.05

SIZE_GCP_TARGET = 2.0
SIZE_GCP_THICKNESS = 0.05
SIZE_GCP_COMPUTED_ARROW = 0.1
SIZE_GCP_RESIDUAL_LINE = 0.02

SIZE_2D_POINT_GCP = 5.0
SIZE_2D_POINT_FEATURE = 2.0

SIZE_LABEL_SHIFT_SHOT = 0.5

SIZE_MATCHGRAPH_LINE = 0.1
SIZE_CAMERA_PATH_LINE = 0.5

MAX_IMAGE_WIDTH = 1500

MATCHGRAPH_CMAP = cm.get_cmap("plasma")
COVERAGE_CMAP = cm.get_cmap("viridis")

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

    # Create a Spatial3D view to display the scene
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/WORLD",
            name="3D Scene",
            background=[13, 17, 23], # Deep dark gray
            line_grid=rrb.LineGrid3D(
                visible=False,
            ),
            spatial_information=rrb.SpatialInformation(
                show_axes=False,
                show_bounding_box=False,
            ),
        ),
        collapse_panels=False,
        )

    rr.send_blueprint(blueprint)

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
    _export_shots(data, reconstruction, _compute_image_plane_distance(reconstruction))
    _export_camera_path(reconstruction)

    # Export 3D points
    _export_points(reconstruction)

    # Export Ground Control Points
    gcp = data.load_ground_control_points()
    if gcp:
        _export_gcp(data, reconstruction, gcp)

    # Export tracks manager (optional, for visualizing observations)
    if data.tracks_exists():
        tracks_manager = data.load_tracks_manager()
        # _export_observations(reconstruction, tracks_manager)
        _export_matchgraph(data, reconstruction, tracks_manager)

    _export_coverage_map(data, reconstruction)

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


def _compute_image_plane_distance(reconstruction: types.Reconstruction) -> float:
    positions = np.array([shot.pose.get_origin() for shot in reconstruction.shots.values()])
    if len(positions) < 2:
        return 2.0
        
    try:
        from scipy.spatial import KDTree
        tree = KDTree(positions)
        dists, _ = tree.query(positions, k=5)
        median_nn_dist = np.median(dists[:, 1])
        return float(median_nn_dist)
    except ImportError:
        logger.warning("Scipy not found, using default image plane distance.")
        return 2.0


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


def _export_shots(data: DataSet, reconstruction: types.Reconstruction, image_plane_distance: float) -> None:
    """Export camera shots (poses and images)."""
    reference = reconstruction.reference

    for shot_id, shot in reconstruction.shots.items():
        _export_shot_pose(shot, shot_id)

        # Collect GPS position and error line if available
        if reference and shot.metadata and shot.metadata.gps_position.has_value:
            gps_topo = shot.metadata.gps_position.value
            _export_shot_gps(shot, shot_id, gps_topo)

        _export_shot_pinhole(data, shot, shot_id, image_plane_distance)


def _export_camera_path(reconstruction: types.Reconstruction) -> None:
    """Export camera path based on capture time."""
    shots_with_time = []
    for shot in reconstruction.shots.values():
        if shot.metadata and shot.metadata.capture_time.has_value:
            shots_with_time.append(shot)
    
    if len(shots_with_time) < 2:
        return

    # Sort by capture time
    shots_with_time.sort(key=lambda x: x.metadata.capture_time.value)
    
    points = [shot.pose.get_origin() for shot in shots_with_time]
    
    rr.log(
        "WORLD/PATH",
        rr.LineStrips3D(
            [points],
            colors=[[255, 255, 255]], # White
            radii=SIZE_CAMERA_PATH_LINE,
        ),
        static=True,
    )
    logger.info(f"Exported camera path with {len(points)} points")


def _export_shot_pose(shot: pymap.Shot, shot_id: str) -> None:
    pose = shot.pose
    R = pose.get_rotation_matrix()  # 3x3 rotation matrix
    t = pose.get_origin()  # 3D position (camera center in world coords)

    # Rerun expects rotation from world to camera
    # OpenSfM GetRotationMatrix gives camera-to-world rotation
    R_world_from_camera = R.T  # Transpose to get world-from-camera

    # Log camera pose with integer sequence ID
    sequence_id = _get_shot_sequence_id(shot_id)
    rr.set_time_sequence("shot", sequence_id)
    rr.log(
        f"WORLD/SHOTS/{shot_id}",
        rr.Transform3D(
            translation=t,
            mat3x3=R_world_from_camera,
            from_parent=False,
        ),
    )

    labels_shift = np.array([0, 0, SIZE_LABEL_SHIFT_SHOT])  # Shift labels above targets
    rr.log(
        f"WORLD/SHOTS/{shot_id}",
        rr.Points3D(
            positions=labels_shift,
            labels=[shot_id],
            radii=0.0,  # Invisible points, just for labels
            colors=[LABEL_COLOR],  # White labels
        ),
        static=True,
    )


def _export_shot_pinhole(data: DataSet, shot: pymap.Shot, shot_id: str, image_plane_distance: float) -> None:
    # Create pinhole camera for image projection
    width = int(shot.camera.width)
    height = int(shot.camera.height)
    width, height = _get_scaled_dimensions(width, height)
    fx, fy, cx, cy = _get_camera_calibration(shot.camera, width, height)

    rr.log(
        f"WORLD/SHOTS/{shot_id}/CAMERA",
        rr.Pinhole(
            resolution=[width, height],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            image_plane_distance=image_plane_distance,
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
            f"WORLD/SHOTS/{shot_id}/CAMERA/IMAGE",
            rr.Image(image).compress(jpeg_quality=50),
        )
    except Exception as e:
        logger.warning(f"Could not load image for {shot_id}: {e}")


def _export_shot_gps(
    shot: pymap.Shot,
    shot_id: str,
    gps_topo: np.ndarray,
) -> None:

    base_path = f"WORLD/GPS/{shot_id}/"

    arrow_size = SIZE_GPS_ARROW
    arrow_radii = arrow_size / 8.0
    arrow_length = arrow_size  # Length of arrows for GPS positions
    
    origin = gps_topo.copy()
    origin[2] += arrow_length
    vector = np.array([0, 0, -arrow_length])

    rr.log(
        f"{base_path}/POSITION",
        rr.Arrows3D(
            origins=[origin],
            vectors=[vector],
            colors=[GPS_COLOR],  # Blue
            radii=arrow_radii,
        ),
        static=True,
    )

    rr.log(
        f"{base_path}/RESIDUAL",
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
        "WORLD/POINTS",
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
    base_path = f"WORLD/GCP/{gcp_id}"

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
        f"{base_path}/TARGET",
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
        f"{base_path}/OUTLINE",
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
        f"{base_path}",
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
            f"{base_path}/COMPUTED",
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
            f"{base_path}/RESIDUAL",
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


def _export_matchgraph(
    data: DataSet,
    reconstruction: types.Reconstruction,
    tracks_manager: pymap.TracksManager,
) -> None:
    """Export match graph connectivity."""
    logger.info("Exporting match graph...")
    
    all_shots = list(reconstruction.shots.keys())
    all_points = list(reconstruction.points.keys())
    
    # Compute connectivity
    connectivity = tracks_manager.get_all_pairs_connectivity(all_shots, all_points)
    if not connectivity:
        return

    all_values = list(connectivity.values())
    if not all_values:
        return
        
    lowest = np.percentile(all_values, 5)
    highest = np.percentile(all_values, 95)
    
    lines = []
    colors = []

    min_inliers = data.config.get("resection_min_inliers", 15)

    for (node1, node2), edge in connectivity.items():
        if edge < 5 * min_inliers:
            continue
            
        if node1 not in reconstruction.shots or node2 not in reconstruction.shots:
            continue
            
        shot1 = reconstruction.shots[node1]
        shot2 = reconstruction.shots[node2]
        
        p1 = shot1.pose.get_origin()
        p2 = shot2.pose.get_origin()
        
        # Normalize edge weight for color mapping
        c_val = max(0.0, min(1.0, 1.0 - (float(edge) - lowest) / (highest - lowest + 1e-6)))
        rgba = MATCHGRAPH_CMAP(1.0 - c_val)
        
        lines.append([p1, p2])
        colors.append([int(c * 255) for c in rgba[:3]])

    if lines:
        rr.log(
            "WORLD/STATS/MATCHGRAPH",
            rr.LineStrips3D(
                lines,
                colors=colors,
                radii=SIZE_MATCHGRAPH_LINE,
            ),
            static=True,
        )
        logger.info(f"Exported match graph with {len(lines)} edges")


def _export_coverage_map(
    data: DataSet,
    reconstruction: types.Reconstruction,
) -> None:
    """Export coverage map as a tessellated quad."""
    logger.info("Exporting coverage map...")
    
    if not reconstruction.points:
        return

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        logger.warning("Scipy not found, skipping coverage map triangulation.")
        return

    # 1. Compute Bounding Box
    points = np.array([p.coordinates for p in reconstruction.points.values()])
    if len(points) == 0:
        return
    
    min_pt = np.percentile(points, 1, axis=0)
    max_pt = np.percentile(points, 99, axis=0)
    median_z = np.median(points[:, 2])

    # Add some margin
    margin_ratio = 0.1
    extent = max_pt - min_pt
    min_pt -= extent * margin_ratio
    max_pt += extent * margin_ratio
    
    # 2. Generate Vertices from Frustums
    vertices = []
    
    for shot in reconstruction.shots.values():
        w = shot.camera.width
        h = shot.camera.height
        
        m = 0.01
        n_steps = 10
        steps = np.linspace(0, 1, n_steps)
        steps = m + steps * (1 - 2 * m)
        
        xs = w * steps
        ys = h * steps
        pixels = features.normalized_image_coordinates(np.array([[x, y] for y in ys for x in xs]), w, h)
        
        # Transform bearings to world frame
        # pose is World -> Camera. Rotation part R.
        bearings = shot.camera.pixel_bearing_many(pixels)
        R_wc = shot.pose.get_rotation_matrix().T
        bearings_world = bearings @ R_wc.T
        
        origin = shot.pose.get_origin()
        
        for direction in bearings_world:
            # Intersect with Z = median_z
            # P = O + t * D
            # median_z = O_z + t * D_z  => t = (median_z - O_z) / D_z
            
            if abs(direction[2]) < 1e-6:
                continue
                
            t = (median_z - origin[2]) / direction[2]
            if t <= 0: # Behind camera or camera is inside plane
                continue
                
            p_ground = origin + t * direction
            
            # Check bounds
            if (p_ground[0] >= min_pt[0] and p_ground[0] <= max_pt[0] and
                p_ground[1] >= min_pt[1] and p_ground[1] <= max_pt[1]):
                vertices.append(p_ground)

    if not vertices:
        logger.warning("No frustum intersections found within bounds.")
        return

    # Remove duplicates/close points to avoid Delaunay issues
    #vertices = np.unique(np.round(np.array(vertices), 3), axis=0)
    
    if len(vertices) < 3:
        return

    # 3. Delaunay Triangulation
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, :2])
    indices = tri.simplices
    
    # 4. Compute Visibility
    counts = np.zeros(len(vertices), dtype=int)
    
    for shot in reconstruction.shots.values():
        # Transform to camera frame
        p_cam = shot.pose.transform_many(vertices)
        
        # Check if in front of camera (Z > 0)
        valid_z = p_cam[:, 2] > 0
        indices_valid_z = np.where(valid_z)[0]
        
        if len(indices_valid_z) == 0:
            continue
            
        p_cam_valid = p_cam[indices_valid_z]
        projections = shot.camera.project_many(p_cam_valid)
        
        # Convert to pixels
        w = shot.camera.width
        h = shot.camera.height
        normalizer = max(w, h)
        
        px = projections[:, 0] * normalizer + w / 2.0
        py = projections[:, 1] * normalizer + h / 2.0
        
        # Check bounds
        in_view = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        
        # Update counts
        counts[indices_valid_z[in_view]] += 1
    
    # 5. Colorize
    # Cap between 2 and 5
    min_count = 0.0
    max_count = 20.0
    
    norm_counts = np.clip(counts, min_count, max_count)
    norm_counts = (norm_counts - min_count) / (max_count - min_count)
    
    colors = COVERAGE_CMAP(norm_counts) # RGBA
    vertex_colors = (colors[:, :3] * 255).astype(np.uint8)

    # 6. Log
    rr.log(
        "WORLD/STATS/COVERAGE",
        rr.Mesh3D(
            vertex_positions=vertices,
            vertex_colors=vertex_colors,
            triangle_indices=indices,
        ),
        static=True
    )
    logger.info(f"Exported coverage map with {len(vertices)} vertices")