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
    Modifies a wall polygon so the observer lies within it. Removes the closest slice
    intersecting the observer safety box and replaces it with two square points farthest
    from the removed wall segment.

    Parameters:
        polygon (array-like): (N, 2) array of (R, Z) coordinates defining the wall.
        observer_pos (tuple): (R_obs, Z_obs) position of the observer.
        safety_distance (float): Half-width of the square region around the observer.

    Returns:
        np.ndarray: Modified (M, 2) wall polygon.
    """
    polygon = np.asarray(polygon)
    R_obs, Z_obs = observer_pos
    R, Z = polygon[:, 0], polygon[:, 1]

    # 1. Create square around observer
    square = np.array([
        [R_obs + safety_distance, Z_obs + safety_distance],
        [R_obs + safety_distance, Z_obs - safety_distance],
        [R_obs - safety_distance, Z_obs - safety_distance],
        [R_obs - safety_distance, Z_obs + safety_distance]
    ])

    # 2. Create masks
    mask_R = (R > R_obs - safety_distance) & (R < R_obs + safety_distance)
    mask_Z = (Z > Z_obs - safety_distance) & (Z < Z_obs + safety_distance)

    # 3. Determine active mask
    if np.any(mask_R) and not np.any(mask_Z):
        active_mask = mask_R
    elif np.any(mask_Z) and not np.any(mask_R):
        active_mask = mask_Z
    elif np.any(mask_R) and np.any(mask_Z):
        # Choose axis with closer points
        dist_R = np.min(np.abs(R[mask_R] - R_obs))
        dist_Z = np.min(np.abs(Z[mask_Z] - Z_obs))
        active_mask = mask_R if dist_R < dist_Z else mask_Z
    else:
        print("Warning: No polygon points near observer. Returning original.")
        return polygon

    # 4. Find continuous slices
    idx = np.where(active_mask)[0]
    if len(idx) == 0:
        print("Warning: No matching points found in mask.")
        return polygon

    slices = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            slices.append((start, idx[i - 1] + 1))
            start = idx[i]
    slices.append((start, idx[-1] + 1))

    # 5. Choose slice closest to observer
    min_dist = np.inf
    best_slice = None
    for s in slices:
        seg = polygon[s[0]:s[1]]
        dist = np.min(np.linalg.norm(seg - np.array([R_obs, Z_obs]), axis=1))
        if dist < min_dist:
            min_dist = dist
            best_slice = s

    if best_slice is None:
        print("Warning: Could not identify slice to remove.")
        return polygon

    start, end = best_slice
    removed_segment = polygon[start:end]

    # 6. Choose 2 farthest square points from removed_segment
    dists = np.array([
        np.min(np.linalg.norm(removed_segment - p, axis=1)) for p in square
    ])
    replacement_points = square[np.argsort(dists)[-2:]]

    # 7. Build final polygon
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
