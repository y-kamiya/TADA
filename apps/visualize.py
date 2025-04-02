import os
import sys
import argparse
import open3d as o3d
import time
import glob
from pathlib import Path
import cv2
import numpy as np


class Visualizer:
    def __init__(self, args):
        self.args = args
        self.obj_files = [Path(dir).absolute() / "mesh.obj" for dir in sorted(glob.glob(f"{args.srcdir}/*"))]
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1280, height=720)

    def fix_mtl(self, obj_path: Path):
        obj_path = obj_path.absolute()
        mtl_path = Path(str(obj_path).replace(".obj", ".mtl"))

        with open(mtl_path, "r") as f:
            lines = f.readlines()

        fixed_mtl = []
        for line in lines:
            if line.startswith("map_Kd "):
                texture_filename = line.split("map_Kd ")[1].strip()
                texture_abs_path = obj_path.parent / texture_filename
                fixed_mtl.append(f"map_Kd {texture_abs_path}\n")
            else:
                fixed_mtl.append(line)

        orig_mtl_path = mtl_path.with_suffix('.orig.mtl')
        mtl_path.replace(orig_mtl_path)

        with open(mtl_path, "w") as f:
            f.writelines(fixed_mtl)

    def set_camera_view(self):
        ctr = self.vis.get_view_control()
        ctr.set_front(args.camera_front)
        ctr.set_zoom(args.camera_zoom)
        
    def run(self):
        if not self.obj_files[0].with_suffix(".orig.mtl").exists():
            self.fix_mtl(self.obj_files[0])

        mesh = o3d.io.read_triangle_mesh(self.obj_files[0], enable_post_processing=True)
        self.vis.add_geometry(mesh)
        self.set_camera_view()

        h, w, _ = np.array(self.vis.capture_screen_float_buffer(do_render=True)).shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        x, y, z = args.camera_front
        output_path = args.srcdir / f"output_zoom{args.camera_zoom}_x{x}_y{y}_z{z}_fps{args.fps}.mp4"
        out = cv2.VideoWriter(output_path, fourcc, self.args.fps, (w, h))

        for obj_file in self.obj_files:
            new_mesh = o3d.io.read_triangle_mesh(obj_file)
            new_mesh.compute_vertex_normals()
            mesh.vertices = new_mesh.vertices
            mesh.triangles = new_mesh.triangles
            mesh.vertex_normals = new_mesh.vertex_normals

            self.vis.update_geometry(mesh)
            self.vis.poll_events()
            self.vis.update_renderer()

            image = self.vis.capture_screen_float_buffer(do_render=True)
            image = (np.asarray(image) * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image)

            time.sleep(1 / self.args.fps)

        out.release()
        self.vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcdir', type=Path, default="out/single576_anneal_ep5-mdm")
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--camera_front', type=float, nargs="*", default=[1, 0.5, 1.5])
    parser.add_argument('--camera_zoom', type=float, default=2.0)
    args = parser.parse_args()
    print(args)

    visualizer = Visualizer(args)
    visualizer.run()

