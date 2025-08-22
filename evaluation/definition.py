import math
import torch

joint_names = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
]

joints_limits ={
    "pelvis_tilt": [-math.pi, math.pi],
    "pelvis_list": [-math.pi, math.pi],
    "pelvis_rotation": [-math.pi, math.pi],
    "pelvis_tx": [-40, 40],
    "pelvis_ty": [-5, 5],
    "pelvis_tz": [-40, 40],
    "hip_flexion_r": [-math.pi/6, 2/3*math.pi], # -30, 120
    "hip_adduction_r": [-5/18*math.pi, math.pi/6], # -50, 30
    "hip_rotation_r": [-2/9*math.pi, 2/9*math.pi], # -40, 40
    "knee_angle_r": [0, 2/3*math.pi], # 0, 120
    "ankle_angle_r": [-2/9*math.pi, math.pi/6], # -40, 30
    "subtalar_angle_r": [-math.pi/9, math.pi/9], # -20, 20
    "mtp_angle_r": [-math.pi/6, math.pi/6], # -30, 30
    "hip_flexion_l": [-math.pi/6, 2/3*math.pi], # -30, 120
    "hip_adduction_l": [-5/18*math.pi, math.pi/6], # -50, 30
    "hip_rotation_l": [-2/9*math.pi, 2/9*math.pi], # -40, 40
    "knee_angle_l": [0, 2/3*math.pi], # 0, 120
    "ankle_angle_l": [-2/9*math.pi, math.pi/6], # -40, 30
    "subtalar_angle_l": [-math.pi/9, math.pi/9], # -20, 20
    "mtp_angle_l": [-math.pi/6, math.pi/6], # -30, 30
    "lumbar_extension": [-math.pi/2, math.pi/2], # -90, 90
    "lumbar_bending": [-math.pi/2, math.pi/2], # -90, 90
    "lumbar_rotation": [-math.pi/2, math.pi/2], # -90, 90
}
pose_param_name2qid = {name: qid for qid, name in enumerate(joint_names)}
qid2pose_param_name = {qid: name for qid, name in enumerate(joint_names)}

OPENSIM_LIM_QIDS = []
OPENSIM_LIM_BOUNDS = []
for name, (lower, upper) in joints_limits.items():
    if lower > upper:
        lower, upper = upper, lower
    OPENSIM_LIM_QIDS.append(pose_param_name2qid[name])
    OPENSIM_LIM_BOUNDS.append([lower, upper])

OPENSIM_LIM_BOUNDS = torch.tensor(OPENSIM_LIM_BOUNDS).float()
OPENSIM_LIM_QID2IDX = {qid: idx for idx, qid in enumerate(OPENSIM_LIM_QIDS)} # inverse mapping