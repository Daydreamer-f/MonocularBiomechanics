from utils import *

def compute_joint_velocities_and_accelerations(mot_reader: MOTReader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Compute joint linear and angular velocities and accelerations
    
    Args:
        mot_reader: MOT file reader
        
    Returns:
        angular_velocities: angular velocity array, shape: (n_frames-1, n_joints) [rad/s]
        angular_accelerations: angular acceleration array, shape: (n_frames-2, n_joints) [rad/s²]
        linear_velocities: linear velocity array, shape: (n_frames-1, n_bodies, 3) [m/s]
        linear_accelerations: linear acceleration array, shape: (n_frames-2, n_bodies, 3) [m/s²]
        joint_names: joint names
        body_names: body names
    """
    joint_data = mot_reader.get_joint_angles()  # shape: (n_frames, n_joints)
    time_data = mot_reader.get_time_data()
    fk = mot_reader.fk
    
    if len(joint_data) < 3:
        raise ValueError("Need at least 3 frames to compute velocities and accelerations")
    
    print("Computing joint velocities and accelerations...")
    
    # compute time interval
    dt = np.diff(time_data)  # shape: (n_frames-1,)
    avg_dt = np.mean(dt)
    
    # 1. compute angular velocity (angle change / time interval)
    angle_changes = np.diff(joint_data, axis=0)  # shape: (n_frames-1, n_joints)
    
    # process angle periodicity
    joint_names = mot_reader.get_joint_names()
    position_joints = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    for i, joint_name in enumerate(joint_names):
        if joint_name not in position_joints:
            # process angle parameters with periodicity
            angle_diff = angle_changes[:, i]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            angle_changes[:, i] = angle_diff
    
    # angular velocity = angle change / time interval
    angular_velocities = angle_changes / dt[:, np.newaxis]  # shape: (n_frames-1, n_joints)
    
    # 2. compute angular acceleration (angular velocity change / time interval)
    angular_velocity_changes = np.diff(angular_velocities, axis=0)  # shape: (n_frames-2, n_joints)
    dt_for_accel = dt[1:]  # time interval for acceleration calculation
    angular_accelerations = angular_velocity_changes / dt_for_accel[:, np.newaxis]  # shape: (n_frames-2, n_joints)
    
    print(f"Angular velocities shape: {angular_velocities.shape}")
    print(f"Angular accelerations shape: {angular_accelerations.shape}")
    
    # 3. compute linear velocity (through forward kinematics)
    print("Computing forward kinematics for linear velocities...")
    
    # batch compute forward kinematics
    @jax.jit
    def compute_fk_batch(joint_angles_batch):
        """批量计算前向运动学"""
        def single_fk(joint_angles):
            return fk(joint_angles, mot_reader.default_scale, check_constraints=False)
        return jax.vmap(single_fk)(joint_angles_batch)
    
    # get all body positions
    joint_data_jax = jnp.array(joint_data)
    batch_size = 50
    n_frames = len(joint_data)
    all_body_positions = []
    
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        batch_joint_data = joint_data_jax[start_idx:end_idx]
        
        print(f"Processing frames {start_idx} to {end_idx-1} for positions...")
        batch_states = compute_fk_batch(batch_joint_data)
        all_body_positions.append(np.array(batch_states.xpos))
    
    all_body_positions = np.concatenate(all_body_positions, axis=0)  # shape: (n_frames, n_bodies, 3)
    
    # compute position changes and linear velocities
    position_changes = np.diff(all_body_positions, axis=0)  # shape: (n_frames-1, n_bodies, 3)
    linear_velocities = position_changes / dt[:, np.newaxis, np.newaxis]  # shape: (n_frames-1, n_bodies, 3)
    
    # 4. compute linear acceleration (linear velocity change / time interval)
    velocity_changes = np.diff(linear_velocities, axis=0)  # shape: (n_frames-2, n_bodies, 3)
    linear_accelerations = velocity_changes / dt_for_accel[:, np.newaxis, np.newaxis]  # shape: (n_frames-2, n_bodies, 3)
    
    print(f"Linear velocities shape: {linear_velocities.shape}")
    print(f"Linear accelerations shape: {linear_accelerations.shape}")
    
    # body names
    body_names = ["world"] + fk.body_names
    
    print(f"Computed velocities and accelerations:")
    print(f"  - Angular velocity range: {np.rad2deg(np.min(angular_velocities)):.2f}° to {np.rad2deg(np.max(angular_velocities)):.2f}° per second")
    print(f"  - Angular acceleration range: {np.rad2deg(np.min(angular_accelerations)):.2f}° to {np.rad2deg(np.max(angular_accelerations)):.2f}° per second²")
    print(f"  - Linear velocity range: {np.min(linear_velocities)*1000:.2f} to {np.max(linear_velocities)*1000:.2f} mm/s")
    print(f"  - Linear acceleration range: {np.min(linear_accelerations)*1000:.2f} to {np.max(linear_accelerations)*1000:.2f} mm/s²")
    
    return angular_velocities, angular_accelerations, linear_velocities, linear_accelerations, joint_names, body_names


def analyze_motion_dynamics(angular_velocities: np.ndarray,
                           angular_accelerations: np.ndarray,
                           linear_velocities: np.ndarray,
                           linear_accelerations: np.ndarray,
                           joint_names: List[str],
                           body_names: List[str],
                           time_data: np.ndarray) -> Dict:
    """
    Analyze motion dynamics
    
    Args:
        angular_velocities: angular velocity array
        angular_accelerations: angular acceleration array
        linear_velocities: linear velocity array
        linear_accelerations: linear acceleration array
        joint_names: joint names
        body_names: body names
        time_data: time data
        
    Returns:
        dynamics analysis results dictionary
    """
    
    # compute statistics of velocities and accelerations
    results = {
        'angular_velocities': {
            'mean_abs_velocity': np.mean(np.abs(angular_velocities), axis=0),
            'max_abs_velocity': np.max(np.abs(angular_velocities), axis=0),
            'std_velocity': np.std(angular_velocities, axis=0),
            'rms_velocity': np.sqrt(np.mean(angular_velocities**2, axis=0)),
            'joint_names': joint_names
        },
        'angular_accelerations': {
            'mean_abs_acceleration': np.mean(np.abs(angular_accelerations), axis=0),
            'max_abs_acceleration': np.max(np.abs(angular_accelerations), axis=0),
            'std_acceleration': np.std(angular_accelerations, axis=0),
            'rms_acceleration': np.sqrt(np.mean(angular_accelerations**2, axis=0)),
            'joint_names': joint_names
        },
        'linear_velocities': {
            'mean_abs_velocity': np.mean(np.linalg.norm(linear_velocities, axis=2), axis=0),  # mean absolute velocity of each body
            'max_abs_velocity': np.max(np.linalg.norm(linear_velocities, axis=2), axis=0),
            'std_velocity': np.std(np.linalg.norm(linear_velocities, axis=2), axis=0),
            'rms_velocity': np.sqrt(np.mean(np.linalg.norm(linear_velocities, axis=2)**2, axis=0)),
            'body_names': body_names
        },
        'linear_accelerations': {
            'mean_abs_acceleration': np.mean(np.linalg.norm(linear_accelerations, axis=2), axis=0),
            'max_abs_acceleration': np.max(np.linalg.norm(linear_accelerations, axis=2), axis=0),
            'std_acceleration': np.std(np.linalg.norm(linear_accelerations, axis=2), axis=0),
            'rms_acceleration': np.sqrt(np.mean(np.linalg.norm(linear_accelerations, axis=2)**2, axis=0)),
            'body_names': body_names
        },
        'smoothness_metrics': {
            # smoothness metrics: jerk of acceleration
            'angular_jerk_rms': np.sqrt(np.mean(np.diff(angular_accelerations, axis=0)**2, axis=0)),
            'linear_jerk_rms': np.sqrt(np.mean(np.linalg.norm(np.diff(linear_accelerations, axis=0), axis=2)**2, axis=0)),
            
            # velocity-acceleration correlation
            'velocity_acceleration_correlation': [
                np.corrcoef(angular_velocities[:len(angular_accelerations), i], angular_accelerations[:, i])[0, 1]
                for i in range(angular_velocities.shape[1])
            ]
        },
        'timing': {
            'velocity_time_range': time_data[1:len(angular_velocities)+1],
            'acceleration_time_range': time_data[2:len(angular_accelerations)+2]
        }
    }
    
    return results


def print_dynamics_summary(results: Dict):
    """
    print dynamics analysis summary
    
    Args:
        results: dynamics analysis results dictionary
    """
    print("\n" + "="*70)
    print("MOTION DYNAMICS ANALYSIS SUMMARY")
    print("="*70)
    
    # angular velocity statistics
    angular_vel_stats = results['angular_velocities']
    print(f"\nTop 10 joints with highest angular velocities:")
    joint_names = angular_vel_stats['joint_names']
    mean_ang_vel = angular_vel_stats['mean_abs_velocity']
    top_ang_vel_indices = np.argsort(mean_ang_vel)[-10:][::-1]
    
    for i, idx in enumerate(top_ang_vel_indices):
        joint_name = joint_names[idx]
        mean_vel = mean_ang_vel[idx]
        max_vel = angular_vel_stats['max_abs_velocity'][idx]
        rms_vel = angular_vel_stats['rms_velocity'][idx]
        print(f"  {i+1}. {joint_name}: mean={np.rad2deg(mean_vel):.2f}°/s, max={np.rad2deg(max_vel):.2f}°/s, rms={np.rad2deg(rms_vel):.2f}°/s")
    
    # angular acceleration statistics
    angular_acc_stats = results['angular_accelerations']
    print(f"\nTop 10 joints with highest angular accelerations:")
    mean_ang_acc = angular_acc_stats['mean_abs_acceleration']
    top_ang_acc_indices = np.argsort(mean_ang_acc)[-10:][::-1]
    
    for i, idx in enumerate(top_ang_acc_indices):
        joint_name = joint_names[idx]
        mean_acc = mean_ang_acc[idx]
        max_acc = angular_acc_stats['max_abs_acceleration'][idx]
        rms_acc = angular_acc_stats['rms_acceleration'][idx]
        print(f"  {i+1}. {joint_name}: mean={np.rad2deg(mean_acc):.1f}°/s², max={np.rad2deg(max_acc):.1f}°/s², rms={np.rad2deg(rms_acc):.1f}°/s²")
    
    # linear velocity statistics
    linear_vel_stats = results['linear_velocities']
    print(f"\nTop 10 body parts with highest linear velocities:")
    body_names = linear_vel_stats['body_names']
    mean_lin_vel = linear_vel_stats['mean_abs_velocity']
    
    # ensure index does not exceed bounds
    n_bodies = min(len(body_names), len(mean_lin_vel))
    top_lin_vel_indices = np.argsort(mean_lin_vel[:n_bodies])[-min(10, n_bodies):][::-1]
    
    for i, idx in enumerate(top_lin_vel_indices):
        if idx < len(body_names):
            body_name = body_names[idx]
            mean_vel = mean_lin_vel[idx]
            max_vel = linear_vel_stats['max_abs_velocity'][idx]
            rms_vel = linear_vel_stats['rms_velocity'][idx]
            print(f"  {i+1}. {body_name}: mean={mean_vel*1000:.1f}mm/s, max={max_vel*1000:.1f}mm/s, rms={rms_vel*1000:.1f}mm/s")
    
    # linear acceleration statistics
    linear_acc_stats = results['linear_accelerations']
    print(f"\nTop 10 body parts with highest linear accelerations:")
    mean_lin_acc = linear_acc_stats['mean_abs_acceleration']
    top_lin_acc_indices = np.argsort(mean_lin_acc[:n_bodies])[-min(10, n_bodies):][::-1]
    
    for i, idx in enumerate(top_lin_acc_indices):
        if idx < len(body_names):
            body_name = body_names[idx]
            mean_acc = mean_lin_acc[idx]
            max_acc = linear_acc_stats['max_abs_acceleration'][idx]
            rms_acc = linear_acc_stats['rms_acceleration'][idx]
            print(f"  {i+1}. {body_name}: mean={mean_acc*1000:.1f}mm/s², max={max_acc*1000:.1f}mm/s², rms={rms_acc*1000:.1f}mm/s²")
    
    # smoothness metrics
    smoothness = results['smoothness_metrics']
    print(f"\nMotion Smoothness Metrics:")
    print(f"  Angular jerk (RMS): {np.rad2deg(np.mean(smoothness['angular_jerk_rms'])):.1f}°/s³")
    print(f"  Linear jerk (RMS): {np.mean(smoothness['linear_jerk_rms'])*1000:.1f}mm/s³")
    
    # overall statistics
    print(f"\nOverall Motion Statistics:")
    print(f"  Max angular velocity: {np.rad2deg(np.max(mean_ang_vel)):.2f}°/s")
    print(f"  Max angular acceleration: {np.rad2deg(np.max(mean_ang_acc)):.1f}°/s²")
    print(f"  Max linear velocity: {np.max(mean_lin_vel)*1000:.1f}mm/s")
    print(f"  Max linear acceleration: {np.max(mean_lin_acc)*1000:.1f}mm/s²")


# modify main function to include new features
def main():
    """
    main function, including velocity and acceleration analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze motion smoothness and dynamics from MOT files')
    parser.add_argument('--mot_path', type=str, default='results/3_subresults.mot',
                       help='Path to the MOT file to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis results')
    parser.add_argument('--include_dynamics', action='store_true', default=True,
                       help='Include velocity and acceleration analysis')
    
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
        smoothness_results = analyze_motion_smoothness(
            angle_changes, body_pos_changes, joint_names, body_names, mot_reader.get_time_data()
        )
        
        # print smoothness summary
        print_smoothness_summary(smoothness_results)
        
        # if include dynamics analysis
        if args.include_dynamics:
            print("\n" + "="*70)
            print("COMPUTING VELOCITIES AND ACCELERATIONS...")
            print("="*70)
            
            # compute velocities and accelerations
            angular_vel, angular_acc, linear_vel, linear_acc, joint_names_dyn, body_names_dyn = compute_joint_velocities_and_accelerations(mot_reader)
            
            # analyze dynamics
            print("\nAnalyzing motion dynamics...")
            dynamics_results = analyze_motion_dynamics(
                angular_vel, angular_acc, linear_vel, linear_acc, 
                joint_names_dyn, body_names_dyn, mot_reader.get_time_data()
            )
            
            # print dynamics summary
            print_dynamics_summary(dynamics_results)
            
            # save dynamics data
            np.save(os.path.join(args.output_dir, 'angular_velocities.npy'), angular_vel)
            np.save(os.path.join(args.output_dir, 'angular_accelerations.npy'), angular_acc)
            np.save(os.path.join(args.output_dir, 'linear_velocities.npy'), linear_vel)
            np.save(os.path.join(args.output_dir, 'linear_accelerations.npy'), linear_acc)
            
            # combine results
            combined_results = {
                'smoothness': smoothness_results,
                'dynamics': dynamics_results
            }
            
            # save combined results
            output_path = os.path.join(args.output_dir, 'complete_motion_analysis.json')
            save_analysis_results(combined_results, output_path)
            
        else:
            # only save smoothness results
            output_path = os.path.join(args.output_dir, 'motion_analysis_results.json')
            save_analysis_results(smoothness_results, output_path)
        
        # save basic data
        angle_changes_path = os.path.join(args.output_dir, 'angle_changes.npy')
        np.save(angle_changes_path, angle_changes)
        print(f"Angle changes saved to: {angle_changes_path}")
        
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