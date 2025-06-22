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


def modify_wall_polygon_for_observer(polygon, observer_pos, safety_distance=0.1, radius_limit=None):
    """
    Modifies wall contour points to ensure the observer is not inside or too close to the mesh.

    Parameters:
        polygon (list or array): Nx2 array of (R,Z) points defining the wall contour.
        observer_pos (tuple): (R_obs, Z_obs) position of the spectroscopic observer.
        safety_distance (float): Minimum distance required between wall and observer [m].
        radius_limit (float or None): Optional radial cutoff (e.g., only modify points with R > limit).
    
    Returns:
        np.ndarray: Modified polygon with adjusted points.
    """
    polygon = np.array(polygon)
    R_obs, Z_obs = observer_pos
    modified_polygon = []

    for r, z in polygon:
        distance = np.sqrt((r - R_obs)**2 + (z - Z_obs)**2)

        if distance < safety_distance:
            # Compute direction vector away from observer
            dr = r - R_obs
            dz = z - Z_obs
            norm = np.sqrt(dr**2 + dz**2)
            if norm == 0:
                dr, dz = 1.0, 0.0  # arbitrary direction if on top of observer
            else:
                dr /= norm
                dz /= norm

            # Optionally limit which points can move (e.g., R > radius_limit)
            if radius_limit is None or r > radius_limit:
                r_new = R_obs + dr * safety_distance
                z_new = Z_obs + dz * safety_distance
                modified_polygon.append((r_new, z_new))
                continue  # move on to next point

        # If not close or not modified
        modified_polygon.append((r, z))

    return np.array(modified_polygon)


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
