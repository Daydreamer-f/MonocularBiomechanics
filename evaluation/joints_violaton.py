import os
import numpy as np
import pandas as pd
import torch
import argparse
from evaluation.definition import (
    joint_names, 
    joints_limits, 
    pose_param_name2qid, 
    qid2pose_param_name,
    OPENSIM_LIM_QIDS,
    OPENSIM_LIM_BOUNDS,
    OPENSIM_LIM_QID2IDX
)
def load_mot_file(mot_path):
    """
    Load joint angles from MOT file
    
    Args:
        mot_path (str): path to the MOT file
        
    Returns:
        torch.Tensor: joint angles, shape (T, n_joints)
        list: joint names
        np.array: time data
    """
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"MOT file not found: {mot_path}")
    
    # read MOT file
    with open(mot_path, 'r') as f:
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
    
    # extract time column
    time_data = df['time'].values
    
    # extract joint angle data (skip time column)
    joint_columns = column_names[1:]  # all columns except time
    joint_data = df[joint_columns].values
    
    # convert to torch tensor
    joint_tensor = torch.from_numpy(joint_data).float()
    
    print(f"Loaded MOT file: {mot_path}")
    print(f"Data shape: {joint_tensor.shape}")
    print(f"Time range: {time_data[0]:.3f} - {time_data[-1]:.3f} seconds")
    print(f"Joint columns found: {len(joint_columns)}")
    
    return joint_tensor, joint_columns, time_data

def eval_rot_delta(poses, joint_column_names, tol_deg=5):
    """
    Evaluate joint angle violations
    
    Args:
        poses: torch.Tensor, joint angles
        joint_column_names: list, joint names
        tol_deg: float, tolerance (degrees)
        
    Returns:
        dict: violation statistics
    """
    tol_rad = np.deg2rad(tol_deg)
    
    # create mapping from joint name to column index
    column_name_to_idx = {name: idx for idx, name in enumerate(joint_column_names)}
    
    # only process joints with defined limits
    violations_per_joint = {}
    
    for joint_name in joint_names:
        if joint_name not in column_name_to_idx:
            print(f"Warning: Joint '{joint_name}' not found in MOT file columns")
            continue
            
        if joint_name not in joints_limits:
            continue
            
        col_idx = column_name_to_idx[joint_name]
        qid = pose_param_name2qid[joint_name]
        
        if qid not in OPENSIM_LIM_QID2IDX:
            continue
            
        lim_idx = OPENSIM_LIM_QID2IDX[qid]
        lower_bound, upper_bound = OPENSIM_LIM_BOUNDS[lim_idx]
        
        # get the joint angle data
        joint_angles = poses[:, col_idx]
        
        normalized_angles = joint_angles
        
        # calculate violation amount
        exceed_lb = torch.where(
            normalized_angles < lower_bound - tol_rad,
            normalized_angles - lower_bound + tol_rad, 
            torch.tensor(0.0)
        )
        exceed_ub = torch.where(
            normalized_angles > upper_bound + tol_rad,
            normalized_angles - upper_bound - tol_rad, 
            torch.tensor(0.0)
        )
        
        violation_amounts = exceed_lb.abs() + exceed_ub.abs()
        violations_per_joint[joint_name] = violation_amounts
    
    return violations_per_joint

def analyze_motion_sequence(poses, joint_column_names, time_data):
    """
    Analyze motion sequence of Euler angles and violations
    
    Args:
        poses: torch.Tensor, shape = (T, n_joints)
        joint_column_names: list, joint names
        time_data: np.array, time data
        
    Returns:
        dict: dictionary containing Euler angles and violation information
    """
    T = poses.shape[0]
    
    # check violations (zero tolerance)
    violations = eval_rot_delta(poses, joint_column_names, tol_deg=0)
    
    # create mapping from joint name to column index
    column_name_to_idx = {name: idx for idx, name in enumerate(joint_column_names)}
    
    # detailed violation analysis (for all joints)
    detailed_violations = {}
    for t in range(T):
        frame_violations = {}
        
        for joint_name in joint_names:
            if joint_name not in column_name_to_idx:
                continue
                
            if joint_name not in joints_limits:
                continue
                
            col_idx = column_name_to_idx[joint_name]
            angle = poses[t, col_idx].item()
            
            # get limits
            lower_bound, upper_bound = joints_limits[joint_name]
            
            # normalize angle
            if joint_name in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                # position parameters do not need angle normalization
                angle_norm = angle
            else:
                # angle parameters need normalization
                angle_norm = (angle + np.pi) % (2 * np.pi) - np.pi
            
            # calculate violation amount
            violation_amount = 0
            if angle_norm < lower_bound:
                violation_amount = lower_bound - angle_norm
            elif angle_norm > upper_bound:
                violation_amount = angle_norm - upper_bound
            
            frame_violations[joint_name] = {
                'angle': float(angle_norm),
                'bounds': [float(lower_bound), float(upper_bound)],
                'violation': float(violation_amount),
                'is_violated': violation_amount > 0
            }
        
        detailed_violations[t] = frame_violations
    
    return {
        'violations': violations,
        'detailed_violations': detailed_violations,
        'time_data': time_data
    }

def print_violation_summary(results):
    """
    print violation summary
    """
    detailed_violations = results['detailed_violations']
    time_data = results['time_data']
    
    print("\n" + "="*60)
    print("JOINT VIOLATION ANALYSIS SUMMARY")
    print("="*60)
    
    # count violations per joint
    joint_violation_counts = {}
    total_violations = 0
    
    for frame_id, frame_violations in detailed_violations.items():
        for joint_name, info in frame_violations.items():
            if info['is_violated']:
                if joint_name not in joint_violation_counts:
                    joint_violation_counts[joint_name] = 0
                joint_violation_counts[joint_name] += 1
                total_violations += 1
    
    print(f"Total frames analyzed: {len(detailed_violations)}")
    print(f"Total violations detected: {total_violations}")
    print(f"Joints with violations: {len(joint_violation_counts)}")
    
    if joint_violation_counts:
        print("\nViolations per joint:")
        for joint_name, count in sorted(joint_violation_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(detailed_violations)) * 100
            print(f"  {joint_name}: {count} violations ({percentage:.1f}% of frames)")
    
    # show top 10 frames with most violations
    print("\nTop 10 frames with most violations:")
    frame_violation_counts = []
    for frame_id, frame_violations in detailed_violations.items():
        violated_count = sum(1 for info in frame_violations.values() if info['is_violated'])
        if violated_count > 0:
            frame_violation_counts.append((frame_id, violated_count, time_data[frame_id]))
    
    frame_violation_counts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (frame_id, count, timestamp) in enumerate(frame_violation_counts[:10]):
        violated_joints = [name for name, info in detailed_violations[frame_id].items() if info['is_violated']]
        print(f"  Frame {frame_id} (t={timestamp:.3f}s): {count} violations - {violated_joints}")

def save_violation_report(results, output_path):
    """
    save detailed violation report to file
    """
    detailed_violations = results['detailed_violations']
    time_data = results['time_data']
    
    with open(output_path, 'w') as f:
        f.write("Joint Violation Analysis Report\n")
        f.write("="*50 + "\n\n")
        
        for frame_id, frame_violations in detailed_violations.items():
            timestamp = time_data[frame_id]
            f.write(f"Frame {frame_id} (Time: {timestamp:.4f}s)\n")
            f.write("-" * 30 + "\n")
            
            violated_joints = []
            for joint_name, info in frame_violations.items():
                if info['is_violated']:
                    violated_joints.append(joint_name)
                    f.write(f"  VIOLATION - {joint_name}:\n")
                    f.write(f"    Angle: {info['angle']:.4f} rad\n")
                    f.write(f"    Bounds: [{info['bounds'][0]:.4f}, {info['bounds'][1]:.4f}] rad\n")
                    f.write(f"    Violation amount: {info['violation']:.4f} rad\n")
            
            if not violated_joints:
                f.write("  No violations detected.\n")
            
            f.write("\n")
    
    print(f"Detailed violation report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze joint violations in OpenSim MOT files')
    parser.add_argument('--mot_path', type=str, default='results/3_subresults.mot',
                       help='Path to the MOT file to analyze')
    parser.add_argument('--output_report', type=str, default='evaluation/violation_report.txt',
                       help='Path to save the detailed violation report')
    parser.add_argument('--tolerance_deg', type=float, default=0,
                       help='Tolerance in degrees for violation detection')
    
    args = parser.parse_args()
    
    try:
        # load MOT file
        print("Loading MOT file...")
        poses, joint_column_names, time_data = load_mot_file(args.mot_path)
        print(f"Successfully loaded poses with shape: {poses.shape}")
        
        # analyze motion sequence
        print("\nAnalyzing motion sequence...")
        results = analyze_motion_sequence(poses, joint_column_names, time_data)
        
        # print summary
        print_violation_summary(results)
        
        # save detailed report
        os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
        save_violation_report(results, args.output_report)
        
        print(f"\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()