import numpy as np
import wildmeshing as wm
import meshio
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mesh_to_tet import convert_mesh_to_tet


def remesh(filename, stop_quality=10, max_its=50, edge_length_r=0.1, epsilon=0.01, visualize=False, 
           mesh_to_tet=False, output_filename="remeshed", scale_factor=1.):
    """
    Remeshes a 3D triangular surface mesh using wildmeshing. This is useful for
    improving the quality of the mesh, and for ensuring that the mesh is
    watertight. This function first tetrahedralizes the mesh, and then
    extracts the surface mesh from the tetrahedralization. The resulting mesh
    is guaranteed to be watertight, but may have a different topology than the
    input mesh.

    This function requires that wildmeshing is installed, see
    https://wildmeshing.github.io/python/ for installation instructions.

    Args:
        filename: .obj filename containing mesh.
        stop_quality: The maximum AMIPS energy for stopping mesh optimization.
        max_its: The maximum number of mesh optimization iterations.
        edge_length_r: The relative target edge length as a fraction of the bounding box diagonal.
        epsilon: The relative envelope size as a fraction of the bounding box diagonal.
        visualize: If True, visualize the input mesh next to the remeshed result using matplotlib.
        scale_factor: The scale factor to apply to the input mesh before remeshing.
        mesh_to_tet: If True, write the tetrahedralization to a .mesh and .tet file.
    """
    m = meshio.read(filename)
    vertices = np.array(m.points)
    faces = np.array(m.cells[0].data, dtype=np.int32)
    #     import openmesh
    #     m = openmesh.read_trimesh(filename)
    #     vertices = np.array(m.points())
    #     faces = np.array(m.face_vertex_indices(), dtype=np.int32)
   
    tetra = wm.Tetrahedralizer(
        stop_quality=stop_quality, max_its=max_its, edge_length_r=edge_length_r, epsilon=epsilon)
    tetra.set_mesh(vertices, np.array(faces).reshape(-1, 3))
    tetra.tetrahedralize()
    tet_vertices, tet_indices = tetra.get_tet_mesh()
    tet_vertices *= scale_factor

    def face_indices(tet):
        face1 = (tet[0], tet[2], tet[1])
        face2 = (tet[1], tet[2], tet[3])
        face3 = (tet[0], tet[1], tet[3])
        face4 = (tet[0], tet[3], tet[2])
        return (
            (face1, tuple(sorted(face1))),
            (face2, tuple(sorted(face2))),
            (face3, tuple(sorted(face3))),
            (face4, tuple(sorted(face4))))

    # # determine surface faces
    elements_per_face = defaultdict(set)
    unique_faces = {}
    for e, tet in enumerate(tet_indices):
        for face, key in face_indices(tet):
            elements_per_face[key].add(e)
            unique_faces[key] = face
    surface_faces = [face for key, face in unique_faces.items() if len(elements_per_face[key]) == 1]
    
    new_vertices = np.array(tet_vertices)
    new_faces = np.array(surface_faces, dtype=int)  
    if mesh_to_tet:
        new_faces = np.array(tet_indices, dtype=np.int32)
        mesh = meshio.Mesh(new_vertices, {"tetra": new_faces})
        meshio.write(f"{output_filename}.mesh", mesh)
        convert_mesh_to_tet(f"{output_filename}.mesh", f"{output_filename}.tet")

    if visualize:
        # render meshes side by side with matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # scale axes equally
        max_range = np.array([vertices[:, i].max() - vertices[:, i].min() for i in range(3)]).max() / 2.0
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        ax.set_title(f"Original ({len(faces)} faces)")
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor='k')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax = fig.add_subplot(122, projection='3d')
        ax.set_title(f"Remeshed ({len(new_faces)} faces)")
        ax.plot_trisurf(new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2], triangles=new_faces, edgecolor='k')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show()

    return new_vertices, new_faces


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Remesh a mesh using wildmeshing.')
    parser.add_argument('filename', type=str, help='Input mesh filename.')
    parser.add_argument('--stop_quality', type=float, default=10, help='The maximum AMIPS energy for stopping mesh optimization.')
    parser.add_argument('--max_its', type=int, default=50, help='The maximum number of mesh optimization iterations.')
    parser.add_argument('--edge_length_r', type=float, default=0.1, help='The relative target edge length as a fraction of the bounding box diagonal.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='The relative envelope size as a fraction of the bounding box diagonal.')
    parser.add_argument('--visualize', action='store_true', help='If True, visualize the input mesh next to the remeshed result using matplotlib.')
    parser.add_argument("--scale_factor", type=float, default=1.0, help="The scale factor to apply to the input mesh before remeshing.")
    parser.add_argument('--mesh_to_tet', action='store_true', help='If True, convert the input mesh to a tetrahedral mesh using meshio.')
    parser.add_argument('--output_filename', '-o', type=str, default='remeshed', help='Output filename.')
    args = parser.parse_args()

    remesh(args.filename, args.stop_quality, args.max_its, args.edge_length_r, args.epsilon, args.visualize, args.mesh_to_tet, args.output_filename, args.scale_factor)

