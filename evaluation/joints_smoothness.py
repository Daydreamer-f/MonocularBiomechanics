import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
from typing import Tuple, List, Dict, Optional
from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics

class MOTReader:
    """
    MOT file reader, for loading and processing OpenSim MOT format joint angle data
    """
    
    def __init__(self, mot_path: str):
        """
        Initialize MOT reader
        
        Args:
            mot_path: path to the MOT file
        """
        self.mot_path = mot_path
        self.data = None
        self.joint_names = None
        self.time_data = None
        self.joint_data = None
        
        # initialize ForwardKinematics
        self.fk = ForwardKinematics(
            xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
        )
        
        # default scale parameters
        self.default_scale = jnp.ones((self.fk.mjx_model.nbody, 1))
        
        self._load_mot_file()
    
    def _load_mot_file(self):
        """
        Load MOT file data
        """
        if not os.path.exists(self.mot_path):
            raise FileNotFoundError(f"MOT file not found: {self.mot_path}")
        
        # read MOT file
        with open(self.mot_path, 'r') as f:
            lines = f.readlines()
        
        # find the line starting with 'time'
        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('time'):
                data_start_idx = i
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find header line starting with 'time' in MOT file")
        
        # parse header
        header_line = lines[data_start_idx].strip()
        column_names = header_line.split('\t')
        
        # read data part
        data_lines = lines[data_start_idx + 1:]
        data_matrix = []
        
        for line in data_lines:
            line = line.strip()
            if line:  # skip empty lines
                values = [float(x) for x in line.split('\t')]
                data_matrix.append(values)
        
        data_matrix = np.array(data_matrix)
        
        # create DataFrame for easier processing
        df = pd.DataFrame(data_matrix, columns=column_names)
        
        # store data
        self.data = df
        self.time_data = df['time'].values
        self.joint_names = column_names[1:]  # all columns except time
        self.joint_data = df[self.joint_names].values  # shape: (n_frames, n_joints)
        
        print(f"Loaded MOT file: {self.mot_path}")
        print(f"Data shape: {self.joint_data.shape}")
        print(f"Time range: {self.time_data[0]:.3f} - {self.time_data[-1]:.3f} seconds")
        print(f"Number of joints: {len(self.joint_names)}")
    
    def get_joint_angles(self, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Get joint angle data
        
        Args:
            frame_idx: frame index, if None, return all frames
            
        Returns:
            joint angle data (radians)
        """
        if frame_idx is not None:
            return self.joint_data[frame_idx]
        return self.joint_data
    
    def get_time_data(self) -> np.ndarray:
        """
        Get time data
        
        Returns:
            time array
        """
        return self.time_data
    
    def get_joint_names(self) -> List[str]:
        """
        Get joint names
        
        Returns:
            joint names
        """
        return self.joint_names
    
    def get_num_frames(self) -> int:
        """
        Get total number of frames
        
        Returns:
            total number of frames
        """
        return len(self.joint_data)


def compute_joint_angle_changes(mot_reader: MOTReader) -> Tuple[np.ndarray, List[str]]:
    """
    Compute angle changes between consecutive frames for each joint
    
    Args:
        mot_reader: MOT file reader
        
    Returns:
        angle_changes: angle change array, shape: (n_frames-1, n_joints)
        joint_names: joint names
    """
    joint_data = mot_reader.get_joint_angles()  # shape: (n_frames, n_joints)
    joint_names = mot_reader.get_joint_names()
    
    if len(joint_data) < 2:
        raise ValueError("Need at least 2 frames to compute angle changes")
    
    # compute angle differences between consecutive frames
    angle_changes = np.diff(joint_data, axis=0)  # shape: (n_frames-1, n_joints)
    
    # for angle parameters, consider periodicity (-π to π jump)
    position_joints = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']  # position parameters, no angle normalization
    
    for i, joint_name in enumerate(joint_names):
        if joint_name not in position_joints:
            # process angle parameters with periodicity
            angle_diff = angle_changes[:, i]
            # limit angle difference to (-π, π) range
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            angle_changes[:, i] = angle_diff
    
    print(f"Computed angle changes for {len(joint_names)} joints across {len(angle_changes)} frame transitions")
    print(f"Angle changes shape: {angle_changes.shape}")
    
    return angle_changes, joint_names


def compute_joint_position_changes(mot_reader: MOTReader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute 3D position changes between consecutive frames for each joint
    
    Args:
        mot_reader: MOT file reader
        
    Returns:
        position_changes: position change array, shape: (n_frames-1, n_bodies, 3)
        site_position_changes: site position change array, shape: (n_frames-1, n_sites, 3)  
        body_names: body names
    """
    joint_data = mot_reader.get_joint_angles()  # shape: (n_frames, n_joints)
    fk = mot_reader.fk
    
    if len(joint_data) < 2:
        raise ValueError("Need at least 2 frames to compute position changes")
    
    print("Computing forward kinematics for all frames...")
    
    # compute forward kinematics for all frames
    all_body_positions = []
    all_site_positions = []
    
    # batch compute forward kinematics
    @jax.jit
    def compute_fk_batch(joint_angles_batch):
        """batch compute forward kinematics"""
        def single_fk(joint_angles):
            return fk(joint_angles, mot_reader.default_scale, check_constraints=False)
        
        # use vmap for batch computation
        return jax.vmap(single_fk)(joint_angles_batch)
    
    # convert data to JAX array
    joint_data_jax = jnp.array(joint_data)
    
    # batch process to avoid memory issues
    batch_size = 50
    n_frames = len(joint_data)
    
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        batch_joint_data = joint_data_jax[start_idx:end_idx]
        
        print(f"Processing frames {start_idx} to {end_idx-1}...")
        
        # compute forward kinematics for this batch
        batch_states = compute_fk_batch(batch_joint_data)
        
        # extract body positions and site positions
        all_body_positions.append(np.array(batch_states.xpos))
        all_site_positions.append(np.array(batch_states.site_xpos))
    
    # merge all batch results
    all_body_positions = np.concatenate(all_body_positions, axis=0)  # shape: (n_frames, n_bodies, 3)
    all_site_positions = np.concatenate(all_site_positions, axis=0)  # shape: (n_frames, n_sites, 3)
    
    print(f"Body positions shape: {all_body_positions.shape}")
    print(f"Site positions shape: {all_site_positions.shape}")
    
    # compute position changes between consecutive frames
    body_position_changes = np.diff(all_body_positions, axis=0)  # shape: (n_frames-1, n_bodies, 3)
    site_position_changes = np.diff(all_site_positions, axis=0)  # shape: (n_frames-1, n_sites, 3)
    
    print(f"Computed position changes for {all_body_positions.shape[1]} body parts across {len(body_position_changes)} frame transitions")
    print(f"Body position changes shape: {body_position_changes.shape}")
    print(f"Site position changes shape: {site_position_changes.shape}")
    
    return body_position_changes, site_position_changes, fk.body_names


def analyze_motion_smoothness(angle_changes: np.ndarray, 
                             position_changes: np.ndarray,
                             joint_names: List[str],
                             body_names: List[str],
                             time_data: np.ndarray) -> Dict:
    """
    Analyze motion smoothness
    
    Args:
        angle_changes: angle change array
        position_changes: position change array  
        joint_names: joint names
        body_names: body names
        time_data: time data
        
    Returns:
        analysis results dictionary
    """
    # compute time interval
    dt = np.diff(time_data)
    avg_dt = np.mean(dt)
    
    # compute angular and position velocities
    angular_velocities = angle_changes / dt[:, np.newaxis]  # rad/s
    position_velocities = position_changes / dt[:, np.newaxis, np.newaxis]  # m/s
    
    # compute statistics
    results = {
        'angle_changes': {
            'mean_abs_change': np.mean(np.abs(angle_changes), axis=0),
            'max_abs_change': np.max(np.abs(angle_changes), axis=0),
            'std_change': np.std(angle_changes, axis=0),
            'joint_names': joint_names
        },
        'position_changes': {
            'mean_abs_change': np.mean(np.abs(position_changes), axis=(0, 2)),  # average change per body part
            'max_abs_change': np.max(np.abs(position_changes), axis=(0, 2)),
            'std_change': np.std(position_changes, axis=(0, 2)),
            'body_names': body_names
        },
        'velocities': {
            'angular_velocities': angular_velocities,
            'position_velocities': position_velocities,
            'mean_angular_speed': np.mean(np.abs(angular_velocities), axis=0),
            'mean_position_speed': np.mean(np.linalg.norm(position_velocities, axis=2), axis=0)
        },
        'timing': {
            'avg_dt': avg_dt,
            'dt_std': np.std(dt),
            'fps': 1.0 / avg_dt
        }
    }
    
    return results


def print_smoothness_summary(results: Dict):
    """
    print motion smoothness analysis summary
    
    Args:
        results: analysis results dictionary
    """
    print("\n" + "="*60)
    print("MOTION SMOOTHNESS ANALYSIS SUMMARY")
    print("="*60)
    
    # time information
    timing = results['timing']
    print(f"Average time step: {timing['avg_dt']:.4f} s")
    print(f"Estimated FPS: {timing['fps']:.2f}")
    print(f"Time step std: {timing['dt_std']:.6f} s")
    
    # angle change statistics
    angle_stats = results['angle_changes']
    print(f"\nTop 5 joints with largest angle changes:")
    joint_names = angle_stats['joint_names']
    mean_changes = angle_stats['mean_abs_change']
    top_joint_indices = np.argsort(mean_changes)[-5:][::-1]
    
    for i, idx in enumerate(top_joint_indices):
        joint_name = joint_names[idx]
        mean_change = mean_changes[idx]
        max_change = angle_stats['max_abs_change'][idx]
        print(f"  {i+1}. {joint_name}: mean={np.rad2deg(mean_change):.3f}°, max={np.rad2deg(max_change):.3f}°")
    
    # position change statistics
    pos_stats = results['position_changes']
    print(f"\nTop 5 body parts with largest position changes:")
    body_names = pos_stats['body_names']
    mean_pos_changes = pos_stats['mean_abs_change']
    top_body_indices = np.argsort(mean_pos_changes)[-5:][::-1]
    
    for i, idx in enumerate(top_body_indices):
        body_name = body_names[idx]
        mean_change = mean_pos_changes[idx]
        max_change = pos_stats['max_abs_change'][idx]
        print(f"  {i+1}. {body_name}: mean={mean_change*1000:.3f}mm, max={max_change*1000:.3f}mm")
    
    # velocity statistics
    vel_stats = results['velocities']
    print(f"\nVelocity Statistics:")
    print(f"Max angular velocity: {np.rad2deg(np.max(vel_stats['mean_angular_speed'])):.2f}°/s")
    print(f"Max position velocity: {np.max(vel_stats['mean_position_speed'])*1000:.2f}mm/s")


def save_analysis_results(results: Dict, output_path: str):
    """
    save analysis results to file
    
    Args:
        results: analysis results dictionary
        output_path: output file path
    """
    import json
    
    # convert numpy array to list for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    json_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Analysis results saved to: {output_path}")


def main():
    """
    main function example
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze motion smoothness from MOT files')
    parser.add_argument('--mot_path', type=str, default='results/3_subresults.mot',
                       help='Path to the MOT file to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    try:
        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # load MOT file
        print("Loading MOT file...")
        mot_reader = MOTReader(args.mot_path)
        
        # compute angle changes
        print("\nComputing joint angle changes...")
        angle_changes, joint_names = compute_joint_angle_changes(mot_reader)
        
        # compute position changes
        print("\nComputing joint position changes...")
        body_pos_changes, site_pos_changes, body_names = compute_joint_position_changes(mot_reader)
        
        # analyze motion smoothness
        print("\nAnalyzing motion smoothness...")
        results = analyze_motion_smoothness(
            angle_changes, body_pos_changes, joint_names, body_names, mot_reader.get_time_data()
        )
        
        # print summary
        print_smoothness_summary(results)
        
        # save results
        output_path = os.path.join(args.output_dir, 'motion_analysis_results.json')
        save_analysis_results(results, output_path)
        
        # save angle changes data
        angle_changes_path = os.path.join(args.output_dir, 'angle_changes.npy')
        np.save(angle_changes_path, angle_changes)
        print(f"Angle changes saved to: {angle_changes_path}")
        
        # save position changes data
        pos_changes_path = os.path.join(args.output_dir, 'position_changes.npy')
        np.save(pos_changes_path, body_pos_changes)
        print(f"Position changes saved to: {pos_changes_path}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()