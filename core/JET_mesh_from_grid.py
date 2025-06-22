import numpy as np
import matplotlib.pyplot as plt
from raysect.optical import World
from raysect.optical.library.metal import RoughTungsten, RoughBeryllium
from raysect.primitive import Mesh
from cherab.tools.primitives.toroidal_mesh import toroidal_mesh_from_polygon

def create_toroidal_wall_from_points(
    points: np.ndarray, 
    parent,
    tungsten_roughness: float=0.29, # Roughness values used by the cherab/jet module
    beryllium_roughness: float=0.26,
    toroidal_extent: float = 360.0,
    num_toroidal_segments: int = 64
):
    """
    Create a toroidally symmetric wall mesh from RZ points.

    Parameters:
        points (list or array): List of (R, Z) tuples or Nx2 array.
        z_split (float): Z height below which material is tungsten, above is beryllium.
        tungsten_roughness (float): Surface roughness for tungsten.
        beryllium_roughness (float): Surface roughness for beryllium.
        toroidal_extent (float): Toroidal angle in radians (default: 2π for full torus).
        num_toroidal_segments (int): Number of toroidal segments for mesh resolution.
        parent (Node): Parent scene graph node (default: World()).

    Returns:
        MeshVolume: The generated wall mesh.
    """
    # Create toroidal mesh
    mesh = toroidal_mesh_from_polygon(
        polygon=points,
        toroidal_extent=toroidal_extent,
        polygon_triangles=None,  # auto-generate triangulation
        num_toroidal_segments=num_toroidal_segments
    )

    # Build mesh volume and assign materials
    mesh.parent = parent

    # Assign materials per face (optional: same material for all if needed)
    mesh.material = RoughTungsten(tungsten_roughness)

    return mesh


def modify_wall_polygon_for_observer(polygon, observer_pos, safety_distance=0.1):
    """
    Modifies a wall polygon to ensure the observer lies within it by replacing
    the closest obstructing segment with two farthest square points around the observer.

    Parameters:
        polygon (array-like): (N, 2) array of (R, Z) coordinates defining wall contour.
        observer_pos (tuple): (R_obs, Z_obs) coordinates of the observer.
        safety_distance (float): Half-width of the box around observer for safe clearance.

    Returns:
        np.ndarray: Modified polygon as (M, 2) array.
    """
    polygon = np.asarray(polygon)
    R_obs, Z_obs = observer_pos
    R, Z = polygon[:, 0], polygon[:, 1]

    # Build square around observer
    square = np.array([
        [R_obs + safety_distance, Z_obs + safety_distance],
        [R_obs + safety_distance, Z_obs - safety_distance],
        [R_obs - safety_distance, Z_obs - safety_distance],
        [R_obs - safety_distance, Z_obs + safety_distance]
    ])

    # Compute distances from observer to square points
    distances = np.linalg.norm(square - np.array([R_obs, Z_obs]), axis=1)
    farthest_indices = np.argsort(distances)[-2:]  # indices of two farthest points
    replacement_points = square[farthest_indices]

    # Build separate masks
    mask_R = (R > R_obs - safety_distance) & (R < R_obs + safety_distance)
    mask_Z = (Z > Z_obs - safety_distance) & (Z < Z_obs + safety_distance)

    # Choose which mask to use
    if np.any(mask_R) and not np.any(mask_Z):
        active_mask = mask_R
        axis = 'R'
    elif np.any(mask_Z) and not np.any(mask_R):
        active_mask = mask_Z
        axis = 'Z'
    elif np.any(mask_R) and np.any(mask_Z):
        # Both are populated, choose axis with closer points
        dist_R = np.min(np.abs(R[mask_R] - R_obs))
        dist_Z = np.min(np.abs(Z[mask_Z] - Z_obs))
        if dist_R < dist_Z:
            active_mask = mask_R
            axis = 'R'
        else:
            active_mask = mask_Z
            axis = 'Z'
    else:
        print("Warning: No polygon points found near observer — skipping modification.")
        return polygon

    # Find continuous block of active_mask
    indices = np.where(active_mask)[0]
    if len(indices) < 2:
        print("Warning: Not enough points in active mask to replace. Returning original polygon.")
        return polygon

    # Find contiguous segment (assume polygon is ordered)
    start = indices[0]
    end = indices[-1] + 1  # exclusive

    # Replace obstructing segment with farthest square points
    new_polygon = np.concatenate([
        polygon[:start],
        replacement_points,
        polygon[end:]
    ])

    return new_polygon




def plot_wall_modification(original_polygon, modified_polygon, observer_pos, title="Wall Contour Adjustment"):
    """
    Plots original and modified wall contours with observer location for comparison.

    Parameters:
        original_polygon (array-like): Original (R,Z) wall points.
        modified_polygon (array-like): Modified (R,Z) wall points after shifting.
        observer_pos (tuple): (R_obs, Z_obs) position of the observer.
        title (str): Plot title.
    """
    original_polygon = np.array(original_polygon)
    modified_polygon = np.array(modified_polygon)
    
    # Ensure contours are closed
    if not np.allclose(original_polygon[0], original_polygon[-1]):
        original_polygon = np.vstack([original_polygon, original_polygon[0]])
    if not np.allclose(modified_polygon[0], modified_polygon[-1]):
        modified_polygon = np.vstack([modified_polygon, modified_polygon[0]])

    plt.figure(figsize=(8, 6))
    plt.plot(original_polygon[:, 0], original_polygon[:, 1], 'b-o', label="Original Contour")
    plt.plot(modified_polygon[:, 0], modified_polygon[:, 1], 'r--o', label="Modified Contour")
    plt.plot(observer_pos[0], observer_pos[1], 'k*', markersize=12, label="Observer")

    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mesh_mod.png")
    plt.show()
