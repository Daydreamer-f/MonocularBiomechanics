import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics
fk = ForwardKinematics(
    xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml",
)


def convertDataframeToMot(df, out_path, dt, time_column):
    numFrames = df.shape[0]
    out_file = open(out_path, 'w')
    out_file.write('Coordinates\n')
    out_file.write('version=1\n')
    out_file.write(f'nRows={numFrames}\n')
    out_file.write(f'nColumns={len(df.columns)+1}\n')
    out_file.write('inDegrees=no\n\n')
    out_file.write('If the header above contains a line with \'inDegrees\', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n')
    out_file.write('endheader\n')

    out_file.write('time')
    for i in range(len(df.columns)):
        out_file.write('\t' + df.columns[i])
    out_file.write('\n')

    for i in range(numFrames):
        out_file.write(str(round(dt * i + time_column[0], 5)))
        for j in range(len(df.columns)):
            out_file.write('\t' + str(df.iloc[i, j]))
        out_file.write('\n')
    out_file.close()
    print('Missing kinematics exported to ' + out_path)

def load_and_export_mot(trial_name, base_path):
    selected_file = f"{base_path}/{trial_name}_fitted_model.npz"
    sub_name = 'sub' + base_path.split('sub')[-1].replace('/', '')
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None, None
    
    fname = selected_file.replace('_fitted_model.npz', '')
    biomech_file = selected_file
    result_text = ""
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file}\n"
            result_text += f"- Timesteps: {len(data['qpos'])}\n"
            result_text += f"- Joint positions shape: {data['qpos'].shape}\n"
            result_text += f"- Joint velocities shape: {data['qvel'].shape}\n"

            print(result_text)
            print(data['timestamps'])
            timestamps = data['timestamps']
            if len(timestamps) > 1:
                dt = float(timestamps[1] - timestamps[0])
                print("dt", dt) #defalt 1/24
            else:
                dt = 1.0/30.0

            qpos_df = pd.DataFrame(data['qpos'], columns=fk.joint_names)
            qpos_df['knee_angle_r'] = - qpos_df['knee_angle_r']
            qpos_df['knee_angle_l'] = - qpos_df['knee_angle_l']
            
            axis_transform_mat = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
            txtytz = qpos_df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].values
            txtytz_global = axis_transform_mat @ txtytz.T
            qpos_df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']] = txtytz_global.T

            euler_angles = qpos_df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].values
            
            rot_mats = R.from_euler('ZXY', euler_angles).as_matrix()
            transformed_rot_mats = np.einsum('ij,njk->nik', axis_transform_mat, rot_mats)
            transformed_euler = R.from_matrix(transformed_rot_mats).as_euler('ZXY')
            qpos_df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']] = transformed_euler

            # reverse the pelvis rotation if it is negative
            qpos_df['pelvis_rotation'] = qpos_df['pelvis_rotation'].apply(lambda x: -x if x < 0 else x)
            
            convertDataframeToMot(qpos_df, f"{fname}_{sub_name}.mot", dt, data['timestamps'])
            convertDataframeToMot(qpos_df, f"exports/{trial_name}_{sub_name}.mot", dt, data['timestamps'])
            result_text += f"Successfully exported {trial_name}_{sub_name}.mot\n"
    else:
        result_text += f"No biomechanics data found for {fname}\n"
    print(result_text)

if __name__ == "__main__":
    load_and_export_mot("3", "results")