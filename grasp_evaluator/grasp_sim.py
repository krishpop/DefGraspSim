# Copyright (c) 2020 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""GraspEvaluator class to et up and run grasp evaluations."""

import h5py
import numpy as np
import os
import timeit
import xml.etree.ElementTree as ET
import yaml

from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R

from utils import pandafsm
from utils import uniform_sphere
from utils import metrics_features_utils

ACRONYM_ASSET_DIR = f"/juno/u/tylerlum/github_repos/nerf_grasping/assets"
ACRONYM_DATA_DIR = f"/juno/u/tylerlum/github_repos/acronym/data/grasps"

class GraspSim:
    """Simulate selected object, grasp, material params, and evaluation mode."""
    def __init__(self, object_name: str, grasp_ind: int, oris: list, platform_height: int,
                 use_viewer: bool, density: float, friction: float, mode: str, controller: dict,  
                 tags: dict, timeout: dict,
                 dir: dict, franka: dict, write_results: bool, tag: str='', is_acronym: bool = False):
        """Initialize parameters for simulation for the specific grasp and material properties."""

        # Set configs for submodules
        self.dir_cfg = dir
        self.controller_cfg = controller 
        self.franka_cfg = franka
        self.tags = tags
        self.timeout = timeout
        self.use_viewer = use_viewer
        self.platform_height = platform_height

        # Soft object material parameters
        self.is_acronym = is_acronym
        if is_acronym:
            self.object_name = object_name
        else:
            self.object_name = object_name.lower()
        self.grasp_ind = grasp_ind
        self.oris = oris  # Array of [ori_start, ori_end]
        self.density = density
        self.friction = friction
        self.mode = mode.lower()

        # Directories of assets and results
        self.assets_dir = os.path.abspath(self.dir_cfg['assets_dir'])

        self.franka_urdf = os.path.abspath(self.dir_cfg['franka_urdf'])
        self.results_dir = os.path.abspath(self.dir_cfg['results_dir'])
        if is_acronym:
            self.get_grasp_candidates_acronym()
            obj_name = '_'.join(self.object_name.split('_')[:-1])
            self.object_scale = float(self.object_name.split('_')[-1])
            self.object_path = os.path.join(ACRONYM_ASSET_DIR, "objects", "urdf",  obj_name + ".urdf")
        elif object_name == "test":
            obj_name = "Donut_615d0b4bbec771e99a6e43b878d5b335_0.0179108154069622"
            self.object_scale = float(obj_name.split('_')[-1])
            # sets self.grasp_candidate_poses
            self.get_grasp_candidates_acronym(f"{obj_name}.h5")
            obj_name = '_'.join(obj_name.split('_')[:-1])
            self.object_path = os.path.join(ACRONYM_ASSET_DIR, "objects", "urdf",  obj_name + ".urdf")
            self.is_acronym =True
        else:
            self.object_path = os.path.join(self.assets_dir, self.object_name)
            self.object_scale = 1.0
            self.get_grasp_candidates()
        self.write_results = write_results
        self.tag = tag

        # Load candidate grasp and initialize results folder
        self.init_results_folder()

        # Create and set up simulation environment
        self.viewer = None
        self.gym = gymapi.acquire_gym()
        self.sim, self.sim_params = self.create_sim()
        self.create_env()
        self.set_asset_properties()
        self.set_camera()
        self.set_transforms()
        self.get_regular_vectors()
        self.setup_scene()

    def init_results_folder(self):
        """Create folder where results are saved. Returns whether existing results will be kept."""
        folder_name = self.object_name + "_" + self.tags['results_storage_tag']
        # rewrite the object_file_name using f-string replacement
        object_file_name = f"{self.object_name}_{self.density}_{self.mode}_tag{self.tag}_results.h5"
        # object_file_name = self.object_name + "_" + self.density + "_" + self.mode + "_tag" + self.tag + "_results.h5"



        self.h5_file_path = os.path.join(self.results_dir, folder_name, object_file_name)

        if os.path.exists(self.h5_file_path):
            existing_h5 = h5py.File(self.h5_file_path, 'r')
            existing_timed_out = existing_h5['timed_out'][self.grasp_ind,
                                                          self.oris[0]]
            existing_succeeded = True

            if self.mode == "pickup":
                existing_pos_under_gravity_dset = existing_h5[
                    'positions_under_gravity']
                if np.all(existing_pos_under_gravity_dset[self.grasp_ind] == 0):
                    existing_succeeded = False

            if self.mode == "reorient":
                reorientation_meshes_dset = existing_h5['reorientation_meshes']
                if np.all(reorientation_meshes_dset[self.grasp_ind, self.oris[0],
                                                    0] == 0):
                    existing_succeeded = False

            if self.mode == "lin_acc":
                lin_acc_fail_accs_dset = existing_h5['lin_acc_fail_accs']
                if lin_acc_fail_accs_dset[self.grasp_ind, self.oris[0]] == 0.0:
                    existing_succeeded = False

            if self.mode == "ang_acc":
                ang_acc_fail_accs_dset = existing_h5['ang_acc_fail_accs']
                if ang_acc_fail_accs_dset[self.grasp_ind, self.oris[0]] == 0.0:
                    existing_succeeded = False

            if self.mode == "squeeze_no_gravity":
                max_forces_dset = existing_h5["squeeze_no_gravity_max_force"]
                if np.all(max_forces_dset[self.grasp_ind] == 0):
                    existing_succeeded = False

            existing_h5.close()
            if existing_timed_out == 0.0 and existing_succeeded:
                print("Data already exists, returning")
                return True
            else:
                print("Existing data is imperfect, rerunning")
        return False

    def get_grasp_candidates_acronym(self, filepath=None):
        """Load the candidate grasp of interest."""
        from scipy.spatial.transform import Rotation as R
        if filepath is not None:
            acronym_data_filepath  =os.path.join(ACRONYM_DATA_DIR, filepath)
        else:
            acronym_data_filepath = os.path.join(ACRONYM_DATA_DIR, self.object_name + ".h5")
        f = h5py.File(acronym_data_filepath, "r")
        grasps = np.array(f["grasps/transforms"])
        print("loaded acronym grasp candidates", grasps.shape)
        candidate_grasp = grasps[self.grasp_ind]
        q = R.from_matrix(candidate_grasp[:3, :3]).as_quat()
        p = candidate_grasp[:3, 3]
        # compose candidate grasp, xyz-pos, xyzw-quat
        self.grasp_candidate_poses = np.concatenate([p, q]).reshape(1, -1)
        self.num_grasp_poses = grasps.shape[0]
        print("Number of total grasp candidates", self.num_grasp_poses)
        f.close()


    def get_grasp_candidates(self):
        """Load the candidate grasp of interest."""
        grasp_file_name = self.object_name + "_grasps.h5"
        f = h5py.File(os.path.realpath(os.path.join(self.object_path, grasp_file_name)), 'r')
        self.grasp_candidate_poses = f['poses'][self.grasp_ind:self.grasp_ind + 1]
        self.num_grasp_poses = f['poses'].shape[0]
        print("Number of total grasp candidates", self.num_grasp_poses)
        f.close()

    def create_sim(self):
        return self._create_sim_physx()

    def _create_sim_physx(self):
        # only tested with this one
        sim_type = gymapi.SIM_PHYSX

        # configure sim
        sim_params = gymapi.SimParams()
        # sim_params.dt = 1.0 / 60.0
        sim_params.dt = 1.0 / 1500

        # sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        sim_params.physx.solver_type = 1
        sim_params.physx.num_velocity_iterations = 0
        # sim_params.physx.use_gpu = True

        # sim_params.use_gpu_pipeline = True
        sim_params.use_gpu_pipeline = False

        gpu_physics = 0 
        gpu_render = 0
        if not self.use_viewer:
            gpu_render = -1

        return self.gym.create_sim(
            gpu_physics,
            gpu_render,
            sim_type,
            sim_params,
        ), sim_params

    def create_env(self):
        """Set dimensions of environments."""
        self.envs_per_row = 6
        self.env_dim = 0.3
        if self.mode in ["lin_acc", "ang_acc"]:
            self.env_dim = 1.0

        # Define environment as half-cube (half in vertical direction)
        self.env_lower = gymapi.Vec3(-self.env_dim, 0, -self.env_dim)
        self.env_upper = gymapi.Vec3(self.env_dim, self.env_dim, self.env_dim)

    def set_object_parameters(self, asset_file_object, **kwargs):
        """Write object parameters into URDF file."""
        try:
            tree = ET.parse(asset_file_object)
            root = tree.getroot()
            for key, value in kwargs.items():
                for attribute in root.iter(key):
                    attribute.set('value', str(value))
            tree.write(asset_file_object)
            return True
        except BaseException:
            return False

    def set_asset_properties(self):
        """Define asset properties."""
        asset_root = ''
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.armature = 0.0
        asset_options.thickness = 0.0
        asset_options.linear_damping = 1.0  # Linear damping for rigid bodies
        asset_options.angular_damping = 0.0  # Angular damping for rigid bodies
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        asset_options.override_inertia = False
        asset_options.override_com = False

        asset_options.vhacd_params.mode = 0  # 0 = tetrahedron, 1 = voxel, was 1, but 0 fixed issue with xbox360
        asset_options.vhacd_params.resolution = 600000
        asset_options.vhacd_params.max_convex_hulls = 16
        asset_options.vhacd_params.max_num_vertices_per_ch = 128

        # Load Franka and object assets
        asset_file_platform = os.path.join(self.assets_dir, 'platform.urdf')
        if self.is_acronym:
            asset_file_object = self.object_path
        else:
            asset_file_object = os.path.join(self.object_path, "soft_body.urdf")

        # Set object parameters for object material properties
        set_parameter_result = True
        fail_counter = 0
        while set_parameter_result is False and fail_counter < 10:
            try:
                set_parameter_result = self.set_object_parameters(
                    asset_file_object,
                    density=self.density)
            except BaseException:
                fail_counter += 1
                pass

        # Set asset options
        asset_options.fix_base_link = True
        self.asset_handle_franka = self.gym.load_asset(self.sim, asset_root, self.franka_urdf,
                                                       asset_options)

        asset_options.fix_base_link = True
        # asset_options.min_particle_mass = 1e-20
        self.asset_handle_object = self.gym.load_asset(self.sim, asset_root, asset_file_object,
                                                       asset_options)
        rs_props = self.gym.get_asset_rigid_shape_properties(self.asset_handle_object)
        for p in rs_props:
            p.friction = self.friction
            p.torsion_friction = self.friction
            p.restitution = 0.0

        asset_options.fix_base_link = True
        self.asset_handle_platform = self.gym.load_asset(self.sim, asset_root,
                                                         asset_file_platform, asset_options)

    def set_camera(self):
        """Define camera properties and create Viewer object."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080

        if self.use_viewer:
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            camera_target = gymapi.Vec3(0.0, 1.0, 0.0)
            camera_pos = gymapi.Vec3(-0, 1.02, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, camera_pos, camera_target)

    def set_transforms(self):
        """Define transforms to convert between Trimesh and Isaac Gym conventions."""
        self.from_trimesh_transform = gymapi.Transform()
        self.from_trimesh_transform.r = gymapi.Quat(0, 0.7071068, 0,
                                                    0.7071068)
        self.neg_rot_x_transform = gymapi.Transform()
        self.neg_rot_x = gymapi.Quat(0.7071068, 0, 0, -0.7071068)
        self.neg_rot_x_transform.r = self.neg_rot_x

    def get_regular_vectors(self):
        """Get directions of regularly spaced vectors in a unit ball."""
        all_directions, _, _, _ = uniform_sphere.get_uniform_directions_regular(16)
        self.num_directions = len(all_directions)
        self.all_directions = all_directions[self.oris[0]:self.oris[1] + 1]

    def get_height_of_objects(self, tet_file):
        """Return the height of the soft object."""
        mesh_lines = list(open(tet_file, "r"))
        mesh_lines = [line.strip('\n') for line in mesh_lines]
        zs = []
        for ml in mesh_lines:
            sp = ml.split(" ")
            if sp[0] == 'v':
                zs.append(float(sp[3]))
        return 2 * abs(min(zs))

    @staticmethod
    def _get_mesh_path_from_urdf(urdf_path):
        import xml.etree.ElementTree as ET

        # Load the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Find the mesh filename inside the URDF file
        mesh_path = root.find(".//geometry/mesh").get("filename")
        mesh_path = '/'.join(mesh_path.split("/")[1:])
        return mesh_path

    def get_height_of_trimesh_objects(self):
        """Return the height of the soft object."""
        import trimesh
        # basename = os.path.basename(self.object_path)
        # obj_path = os.path.splitext(basename)[0]
        # obj_name, commit = obj_path.split('_')
        asset_path = self.object_path
        mesh_path = os.path.join(ACRONYM_ASSET_DIR, "objects", "meshes", self._get_mesh_path_from_urdf(asset_path))

        mesh = trimesh.load(mesh_path, force="mesh")
        # scale object by trimesh scale transform
        scale_transform = trimesh.transformations.scale_matrix(self.object_scale)
        mesh.apply_transform(scale_transform)
        (_, _, min_z), _ = mesh.bounds
        return 2 * abs(min_z)

    def setup_scene(self):
        """Create environments, Franka actor, and object actor."""
        self.env_handles = []
        self.franka_handles = []
        self.object_handles = []
        self.platform_handles = []
        self.hand_origins = []

        self.env_spread = self.grasp_candidate_poses
        if self.mode.lower() in ["reorient", "lin_acc", "ang_acc"]:
            self.env_spread = self.all_directions

        for i, test_grasp_pose in enumerate(self.env_spread):
            if self.mode.lower() in ["reorient", "lin_acc", "ang_acc"]:
                test_grasp_pose = self.grasp_candidate_poses[0]
                direction = np.array(
                    [self.all_directions[i][1], self.all_directions[i][2],
                     self.all_directions[i][0]])  # Read direction as y-up convention
            else:
                # test_grasp_pose = np.array([0.,0.,0., 0.,0.,0.,1.], dtype="float32")
                test_grasp_pose = self.grasp_candidate_poses[0]
                direction = np.array(
                    [self.all_directions[0][1], self.all_directions[0][2],
                     self.all_directions[0][0]])  # Read direction as y-up convention

            # Create environment
            env_handle = self.gym.create_env(
                self.sim, self.env_lower, self.env_upper, self.envs_per_row)
            self.env_handles.append(env_handle)

            # Define shared pose/collision parameters
            pose = gymapi.Transform()
            grasp_transform = gymapi.Transform()
            grasp_transform.r = gymapi.Quat(test_grasp_pose[4], test_grasp_pose[5],
                                            test_grasp_pose[6], test_grasp_pose[3])

            _, franka_rpy = metrics_features_utils.get_franka_rpy(grasp_transform.r)

            collision_group = i
            collision_filter = 0

            # Create Franka actors
            pose.p = gymapi.Vec3(test_grasp_pose[0], test_grasp_pose[1],
                                 test_grasp_pose[2])
            pose.p = self.neg_rot_x_transform.transform_vector(pose.p)
            pose.p.y += self.platform_height
            print("franka handle initial pose", np.array(pose.p))
            franka_handle = self.gym.create_actor(env_handle, self.asset_handle_franka, pose,
                                                  f"franka_{i}", collision_group, 1)
            self.franka_handles.append(franka_handle)


            curr_joint_positions = self.gym.get_actor_dof_states(
                env_handle, franka_handle, gymapi.STATE_ALL)

            ang_acc_axis = np.array([0., 0., 1.])
            pose_transform = R.from_euler('ZYX', franka_rpy)
            ang_acc_transform = R.align_vectors(np.expand_dims(direction, axis=0),
                                              np.expand_dims(ang_acc_axis,
                                                             axis=0))[0]
            ang_acc_eulers = ang_acc_transform.as_euler('xyz')

            pose_correction = ang_acc_transform.inv() * pose_transform
            pose_correction_euler = pose_correction.as_euler('xyz')

            # Correct for translation offset to match grasp. Allows for one joint to
            # be solely responsible for generating angular acceleration
            q0 = np.array([0., 0., -0.112])
            q0_ = ang_acc_transform.apply(q0)
            disp_offset = q0 - q0_

            curr_joint_positions['pos'] = [
                disp_offset[0], disp_offset[1], disp_offset[2], ang_acc_eulers[2],
                ang_acc_eulers[1], ang_acc_eulers[0], 0., pose_correction_euler[2],
                pose_correction_euler[1], pose_correction_euler[0], 0.0, 0.0, 0.0,
                0, 0.04, 0.04
            ]

            self.hand_origins.append(pose)
            finger_pose = gymapi.Transform()
            finger_pose.p = pose.p

            self.gym.set_actor_dof_states(env_handle, franka_handle,
                                          curr_joint_positions, gymapi.STATE_ALL)

            # Create soft object
            if self.is_acronym:
                height_of_object = self.get_height_of_trimesh_objects()
            else:
                tet_file_name = os.path.join(self.object_path, self.object_name + ".tet")
                height_of_object = self.get_height_of_objects(tet_file_name)

            pose = gymapi.Transform()
            pose.r = self.neg_rot_x_transform.r

            pose.p = self.from_trimesh_transform.transform_vector(
                gymapi.Vec3(0.0, 0.0, 0.0))

            object_height_buffer = 0.001
            if self.mode == "squeeze_no_gravity":
                object_height_buffer = 0.0
            pose.p.y += self.platform_height + object_height_buffer

            object_handle = self.gym.create_actor(env_handle, self.asset_handle_object, pose,
                                                  f"object_{i}", collision_group,
                                                  collision_filter)
            self.gym.set_actor_scale(env_handle, object_handle, self.object_scale)
            self.object_handles.append(object_handle)

            # Create platform
            height_of_platform = 0.005
            pose.p.y -= (height_of_platform + object_height_buffer +
                         + 0.5 * height_of_object)

            if self.mode == "squeeze_no_gravity":
                pose.p.y = 0.5

            platform_handle = self.gym.create_actor(env_handle, self.asset_handle_platform,
                                                    pose, f"platform_{i}",
                                                    collision_group, 1)
            self.platform_handles.append(platform_handle)

            self.gym.set_rigid_body_color(env_handle, franka_handle, 0,
                                          gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(0, 1, 0))

    def run_simulation(self):
        """Perform grasp evaluation."""
        panda_fsms = []
        directions = self.all_directions

        for i in range(len(self.env_handles)):
            if self.mode.lower() in ["reorient", "lin_acc", "ang_acc"]:
                test_grasp_pose = self.grasp_candidate_poses[0]
                directions = self.all_directions[i:i + 1]

            else:
                test_grasp_pose = np.array([0.,0.,0., 1.,0.,0.,0.], dtype="float32")
                test_grasp_pose = self.env_spread[i]

            pure_grasp_transform = gymapi.Transform()
            pure_grasp_transform.r = gymapi.Quat(test_grasp_pose[4],
                                                 test_grasp_pose[5],
                                                 test_grasp_pose[6],
                                                 test_grasp_pose[3])
            grasp_transform = gymapi.Transform()
            grasp_transform.r = self.neg_rot_x * gymapi.Quat(
                test_grasp_pose[4], test_grasp_pose[5], test_grasp_pose[6],
                test_grasp_pose[3])

            panda_fsm = pandafsm.PandaFsm(cfg=self.franka_cfg,
                                          controller_cfg=self.controller_cfg,
                                          gym_handle=self.gym,
                                          sim_handle=self.sim,
                                          env_handles=self.env_handles,
                                          franka_handle=self.franka_handles[i],
                                          object_handle=self.object_handles[i],
                                          platform_handle=self.platform_handles[i],
                                          object_cof=self.friction,
                                          grasp_transform=grasp_transform,
                                          obj_name=self.object_name,
                                          env_id=i,
                                          hand_origin=self.hand_origins[i],
                                          viewer=self.viewer,
                                          envs_per_row=self.envs_per_row,
                                          env_dim=self.env_dim,
                                          youngs=1.,
                                          density=self.density,
                                          directions=np.asarray(directions),
                                          mode=self.mode.lower())
            panda_fsms.append(panda_fsm)

        all_done = False
        loop_start = timeit.default_timer()

        while not all_done:

            # If the simulation is taking too long, declare fail
            if (timeit.default_timer() - loop_start > self.timeout['other_modes']
                    and panda_fsms[i].state not in ['reorient', 'squeeze_no_gravity']) or (
                        timeit.default_timer()
                        - loop_start > self.timeout['squeeze_no_gravity']
                        and panda_fsms[i].state == "squeeze_no_gravity"):
                print("Timed out")
                for i in range(len(self.env_handles)):
                    if panda_fsms[i].state != "done":
                        panda_fsms[i].state = "done"
                        panda_fsms[i].timed_out = True

            for i in range(len(self.env_handles)):
                left_finger_transform = self.gym.get_rigid_transform(
                                        panda_fsms[i].env_handle, panda_fsms[i].left_finger_handle)
                fingertip_pos = np.array([left_finger_transform.p.x,
                                                   left_finger_transform.p.y,
                                                   left_finger_transform.p.z])
                print('left finger position', fingertip_pos)

            all_done = all(panda_fsms[i].state == 'done'
                           for i in range(len(self.env_handles)))

            # self.gym.refresh_particle_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(len(self.env_handles)):
                if panda_fsms[i].state != "done":
                    panda_fsms[i].run_state_machine()

            # Run simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.clear_lines(self.viewer)
            self.gym.step_graphics(self.sim)

            if self.use_viewer:
                self.gym.draw_viewer(self.viewer, self.sim, True)

        # Clean up
        if self.use_viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        print("Finished the simulation", timeit.default_timer() - loop_start)

        if self.write_results:
            print("Writing to", self.h5_file_path)
            metrics_features_utils.write_metrics_to_h5(self.mode, self.grasp_ind, self.oris,
                                                       self.num_grasp_poses, self.num_directions,
                                                       self.h5_file_path, panda_fsms,
                                                       self.cfg['squeeze_no_gravity']['num_dp'])
        return
