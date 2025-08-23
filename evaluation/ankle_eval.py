import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics

class AnkleMotionAnalyzer:
    """
    踝关节运动分析器，专门分析踝关节的角速度和线速度
    """
    
    def __init__(self, mot_path: str):
        """
        初始化踝关节运动分析器
        
        Args:
            mot_path: MOT文件路径
        """
        self.mot_path = mot_path
        self.data = None
        self.joint_names = None
        self.time_data = None
        self.joint_data = None
        
        # 初始化ForwardKinematics
        self.fk = ForwardKinematics(
            xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
        )
        
        # 默认缩放参数
        self.default_scale = jnp.ones((self.fk.mjx_model.nbody, 1))
        
        # 踝关节相关的关节和身体部位
        self.ankle_joint_names = ['ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
                                 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']
        self.ankle_body_names = ['talus_r', 'calcn_r', 'toes_r', 
                                'talus_l', 'calcn_l', 'toes_l']  # 踝关节对应的身体部位
        
        self._load_mot_file()
        self._identify_ankle_indices()
    
    def _load_mot_file(self):
        """
        加载MOT文件数据
        """
        if not os.path.exists(self.mot_path):
            raise FileNotFoundError(f"MOT file not found: {self.mot_path}")
        
        # 读取MOT文件
        with open(self.mot_path, 'r') as f:
            lines = f.readlines()
        
        # 找到数据开始的行
        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('time'):
                data_start_idx = i
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find header line starting with 'time' in MOT file")
        
        # 解析表头
        header_line = lines[data_start_idx].strip()
        column_names = header_line.split('\t')
        
        # 读取数据部分
        data_lines = lines[data_start_idx + 1:]
        data_matrix = []
        
        for line in data_lines:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split('\t')]
                data_matrix.append(values)
        
        data_matrix = np.array(data_matrix)
        
        # 创建DataFrame
        df = pd.DataFrame(data_matrix, columns=column_names)
        
        # 存储数据
        self.data = df
        self.time_data = df['time'].values
        self.joint_names = column_names[1:]
        self.joint_data = df[self.joint_names].values
        
        print(f"Loaded MOT file: {self.mot_path}")
        print(f"Data shape: {self.joint_data.shape}")
        print(f"Time range: {self.time_data[0]:.3f} - {self.time_data[-1]:.3f} seconds")
        print(f"Number of frames: {len(self.joint_data)}")
    
    def _identify_ankle_indices(self):
        """
        识别踝关节在数据中的索引
        """
        self.ankle_joint_indices = {}
        self.ankle_body_indices = {}
        
        # 找到踝关节在joint_names中的索引
        for ankle_joint in self.ankle_joint_names:
            if ankle_joint in self.joint_names:
                self.ankle_joint_indices[ankle_joint] = self.joint_names.index(ankle_joint)
                print(f"Found {ankle_joint} at index {self.ankle_joint_indices[ankle_joint]}")
            else:
                print(f"Warning: {ankle_joint} not found in joint names")
        
        # 找到踝关节相关身体部位在body_names中的索引
        body_names_with_world = ["world"] + self.fk.body_names
        for ankle_body in self.ankle_body_names:
            if ankle_body in body_names_with_world:
                self.ankle_body_indices[ankle_body] = body_names_with_world.index(ankle_body)
                print(f"Found {ankle_body} at body index {self.ankle_body_indices[ankle_body]}")
            else:
                print(f"Warning: {ankle_body} not found in body names")
        
        print(f"Available joint names: {self.joint_names}")
        print(f"Available body names: {body_names_with_world}")
    
    def compute_ankle_angular_velocities(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        计算踝关节角速度
        
        Returns:
            ankle_angular_velocities: 踝关节角速度字典 {joint_name: velocities}
            time_velocity: 对应的时间数组
        """
        if len(self.joint_data) < 2:
            raise ValueError("Need at least 2 frames to compute angular velocities")
        
        print("Computing ankle angular velocities...")
        
        # 计算时间间隔
        dt = np.diff(self.time_data)
        
        ankle_angular_velocities = {}
        
        for ankle_joint, joint_idx in self.ankle_joint_indices.items():
            # 获取踝关节角度数据
            ankle_angles = self.joint_data[:, joint_idx]
            
            # 计算角度变化
            angle_changes = np.diff(ankle_angles)
            
            # 处理角度周期性（-π到π）
            angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
            
            # 计算角速度
            angular_velocity = angle_changes / dt
            
            ankle_angular_velocities[ankle_joint] = angular_velocity
            
            print(f"{ankle_joint} angular velocity range: {np.rad2deg(np.min(angular_velocity)):.2f}° to {np.rad2deg(np.max(angular_velocity)):.2f}° per second")
        
        time_velocity = self.time_data[1:]  # 速度对应的时间点
        
        return ankle_angular_velocities, time_velocity
    
    def compute_ankle_linear_velocities(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        计算踝关节相关身体部位的线速度
        
        Returns:
            ankle_linear_velocities: 踝关节线速度字典 {body_name: velocities}
            time_velocity: 对应的时间数组
        """
        if len(self.joint_data) < 2:
            raise ValueError("Need at least 2 frames to compute linear velocities")
        
        print("Computing ankle linear velocities using forward kinematics...")
        
        # 批量计算前向运动学
        @jax.jit
        def compute_fk_batch(joint_angles_batch):
            def single_fk(joint_angles):
                return self.fk(joint_angles, self.default_scale, check_constraints=False)
            return jax.vmap(single_fk)(joint_angles_batch)
        
        # 获取所有帧的身体位置
        joint_data_jax = jnp.array(self.joint_data)
        batch_size = 50
        n_frames = len(self.joint_data)
        all_body_positions = []
        
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch_joint_data = joint_data_jax[start_idx:end_idx]
            
            print(f"Processing frames {start_idx} to {end_idx-1}...")
            batch_states = compute_fk_batch(batch_joint_data)
            all_body_positions.append(np.array(batch_states.xpos))
        
        all_body_positions = np.concatenate(all_body_positions, axis=0)
        
        # 计算时间间隔
        dt = np.diff(self.time_data)
        
        ankle_linear_velocities = {}
        
        for ankle_body, body_idx in self.ankle_body_indices.items():
            # 获取该身体部位的位置数据
            body_positions = all_body_positions[:, body_idx, :]
            
            # 计算位置变化
            position_changes = np.diff(body_positions, axis=0)
            
            # 计算线速度
            linear_velocity = position_changes / dt[:, np.newaxis]
            
            # 计算速度幅值
            velocity_magnitude = np.linalg.norm(linear_velocity, axis=1)
            
            ankle_linear_velocities[ankle_body] = {
                'velocity_3d': linear_velocity,
                'velocity_magnitude': velocity_magnitude,
                'velocity_x': linear_velocity[:, 0],
                'velocity_y': linear_velocity[:, 1],
                'velocity_z': linear_velocity[:, 2]
            }
            
            print(f"{ankle_body} linear velocity magnitude range: {np.min(velocity_magnitude)*1000:.2f} to {np.max(velocity_magnitude)*1000:.2f} mm/s")
        
        time_velocity = self.time_data[1:]
        
        return ankle_linear_velocities, time_velocity
    
    def print_frame_by_frame_analysis(self, ankle_angular_vel: Dict[str, np.ndarray], 
                                    ankle_linear_vel: Dict[str, np.ndarray], 
                                    time_velocity: np.ndarray,
                                    max_frames: int = 20):
        """
        打印逐帧分析结果
        """
        print("\n" + "="*120)
        print("FRAME-BY-FRAME ANKLE MOTION ANALYSIS")
        print("="*120)
        
        n_frames = min(len(time_velocity), max_frames)
        
        print(f"Showing first {n_frames} frames:")
        print("-" * 120)
        
        # 表头
        header = f"{'Frame':<6} {'Time(s)':<8} {'ankle_r':<10} {'sub_r':<10} {'mtp_r':<10} {'ankle_l':<10} {'sub_l':<10} {'mtp_l':<10} {'talus_r':<10} {'calcn_r':<10} {'toes_r':<10} {'talus_l':<10} {'calcn_l':<10} {'toes_l':<10}"
        print(header)
        print("-" * 120)
        
        for i in range(n_frames):
            frame_num = i + 1
            time = time_velocity[i]
            
            # 角速度 (转换为度/秒)
            ankle_r = np.rad2deg(ankle_angular_vel.get('ankle_angle_r', [0])[i]) if i < len(ankle_angular_vel.get('ankle_angle_r', [])) else 0
            sub_r = np.rad2deg(ankle_angular_vel.get('subtalar_angle_r', [0])[i]) if i < len(ankle_angular_vel.get('subtalar_angle_r', [])) else 0
            mtp_r = np.rad2deg(ankle_angular_vel.get('mtp_angle_r', [0])[i]) if i < len(ankle_angular_vel.get('mtp_angle_r', [])) else 0
            ankle_l = np.rad2deg(ankle_angular_vel.get('ankle_angle_l', [0])[i]) if i < len(ankle_angular_vel.get('ankle_angle_l', [])) else 0
            sub_l = np.rad2deg(ankle_angular_vel.get('subtalar_angle_l', [0])[i]) if i < len(ankle_angular_vel.get('subtalar_angle_l', [])) else 0
            mtp_l = np.rad2deg(ankle_angular_vel.get('mtp_angle_l', [0])[i]) if i < len(ankle_angular_vel.get('mtp_angle_l', [])) else 0
            
            # 线速度幅值 (转换为毫米/秒)
            talus_r_lin = ankle_linear_vel.get('talus_r', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('talus_r', {}).get('velocity_magnitude', [])) else 0
            calcn_r_lin = ankle_linear_vel.get('calcn_r', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('calcn_r', {}).get('velocity_magnitude', [])) else 0
            toes_r_lin = ankle_linear_vel.get('toes_r', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('toes_r', {}).get('velocity_magnitude', [])) else 0
            talus_l_lin = ankle_linear_vel.get('talus_l', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('talus_l', {}).get('velocity_magnitude', [])) else 0
            calcn_l_lin = ankle_linear_vel.get('calcn_l', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('calcn_l', {}).get('velocity_magnitude', [])) else 0
            toes_l_lin = ankle_linear_vel.get('toes_l', {}).get('velocity_magnitude', [0])[i] * 1000 if i < len(ankle_linear_vel.get('toes_l', {}).get('velocity_magnitude', [])) else 0
            
            print(f"{frame_num:<6} {time:<8.3f} {ankle_r:<10.2f} {sub_r:<10.2f} {mtp_r:<10.2f} {ankle_l:<10.2f} {sub_l:<10.2f} {mtp_l:<10.2f} {talus_r_lin:<10.1f} {calcn_r_lin:<10.1f} {toes_r_lin:<10.1f} {talus_l_lin:<10.1f} {calcn_l_lin:<10.1f} {toes_l_lin:<10.1f}")
    
    def save_detailed_results(self, ankle_angular_vel: Dict[str, np.ndarray], 
                            ankle_linear_vel: Dict[str, np.ndarray], 
                            time_velocity: np.ndarray,
                            output_dir: str):
        """
        保存详细的分析结果
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建详细的DataFrame
        detailed_data = {
            'frame': np.arange(1, len(time_velocity) + 1),
            'time': time_velocity
        }
        
        # 添加角速度数据
        for joint_name, ang_vel in ankle_angular_vel.items():
            detailed_data[f'{joint_name}_angular_velocity_deg_per_s'] = np.rad2deg(ang_vel)
            detailed_data[f'{joint_name}_angular_velocity_rad_per_s'] = ang_vel
        
        # 添加线速度数据
        for body_name, lin_vel_data in ankle_linear_vel.items():
            detailed_data[f'{body_name}_linear_velocity_magnitude_mm_per_s'] = lin_vel_data['velocity_magnitude'] * 1000
            detailed_data[f'{body_name}_linear_velocity_magnitude_m_per_s'] = lin_vel_data['velocity_magnitude']
            detailed_data[f'{body_name}_linear_velocity_x_m_per_s'] = lin_vel_data['velocity_x']
            detailed_data[f'{body_name}_linear_velocity_y_m_per_s'] = lin_vel_data['velocity_y']
            detailed_data[f'{body_name}_linear_velocity_z_m_per_s'] = lin_vel_data['velocity_z']
        
        # 创建DataFrame
        df = pd.DataFrame(detailed_data)
        
        # 保存为CSV文件
        csv_path = os.path.join(output_dir, 'ankle_velocities_detailed.csv')
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # 保存为numpy文件
        np.savez(os.path.join(output_dir, 'ankle_velocities_raw.npz'),
                time=time_velocity,
                **{f'{k}_angular_vel': v for k, v in ankle_angular_vel.items()},
                **{f'{k}_linear_vel': v['velocity_3d'] for k, v in ankle_linear_vel.items()})
        
        print(f"Raw numpy data saved to: {os.path.join(output_dir, 'ankle_velocities_raw.npz')}")
    
    def create_velocity_plots(self, ankle_angular_vel: Dict[str, np.ndarray], 
                            ankle_linear_vel: Dict[str, np.ndarray], 
                            time_velocity: np.ndarray,
                            output_dir: str):
        """
        创建速度图表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Ankle Motion Velocity Analysis', fontsize=16)
        
        # 右踝角速度图表
        ax1 = axes[0, 0]
        for joint_name in ['ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']:
            if joint_name in ankle_angular_vel:
                ax1.plot(time_velocity, np.rad2deg(ankle_angular_vel[joint_name]), 
                        label=joint_name.replace('_r', ''), linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (°/s)')
        ax1.set_title('Right Ankle Angular Velocities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 左踝角速度图表
        ax2 = axes[0, 1]
        for joint_name in ['ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']:
            if joint_name in ankle_angular_vel:
                ax2.plot(time_velocity, np.rad2deg(ankle_angular_vel[joint_name]), 
                        label=joint_name.replace('_l', ''), linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (°/s)')
        ax2.set_title('Left Ankle Angular Velocities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 右脚线速度幅值图表
        ax3 = axes[1, 0]
        for body_name in ['talus_r', 'calcn_r', 'toes_r']:
            if body_name in ankle_linear_vel:
                ax3.plot(time_velocity, ankle_linear_vel[body_name]['velocity_magnitude'] * 1000, 
                        label=body_name.replace('_r', ''), linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Linear Velocity Magnitude (mm/s)')
        ax3.set_title('Right Foot Linear Velocity Magnitudes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 左脚线速度幅值图表
        ax4 = axes[1, 1]
        for body_name in ['talus_l', 'calcn_l', 'toes_l']:
            if body_name in ankle_linear_vel:
                ax4.plot(time_velocity, ankle_linear_vel[body_name]['velocity_magnitude'] * 1000, 
                        label=body_name.replace('_l', ''), linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Linear Velocity Magnitude (mm/s)')
        ax4.set_title('Left Foot Linear Velocity Magnitudes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 右脚趾3D线速度分量
        ax5 = axes[2, 0]
        if 'toes_r' in ankle_linear_vel:
            vel_data = ankle_linear_vel['toes_r']
            ax5.plot(time_velocity, vel_data['velocity_x'] * 1000, label='X', linewidth=2)
            ax5.plot(time_velocity, vel_data['velocity_y'] * 1000, label='Y', linewidth=2)
            ax5.plot(time_velocity, vel_data['velocity_z'] * 1000, label='Z', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Linear Velocity (mm/s)')
        ax5.set_title('Right Toes Linear Velocity Components')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 左脚趾3D线速度分量
        ax6 = axes[2, 1]
        if 'toes_l' in ankle_linear_vel:
            vel_data = ankle_linear_vel['toes_l']
            ax6.plot(time_velocity, vel_data['velocity_x'] * 1000, label='X', linewidth=2)
            ax6.plot(time_velocity, vel_data['velocity_y'] * 1000, label='Y', linewidth=2)
            ax6.plot(time_velocity, vel_data['velocity_z'] * 1000, label='Z', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Linear Velocity (mm/s)')
        ax6.set_title('Left Toes Linear Velocity Components')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, 'ankle_velocity_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Velocity plots saved to: {plot_path}")
        
        plt.close()
    
    def print_statistics_summary(self, ankle_angular_vel: Dict[str, np.ndarray], 
                                ankle_linear_vel: Dict[str, np.ndarray]):
        """
        打印统计摘要
        """
        print("\n" + "="*60)
        print("ANKLE MOTION STATISTICS SUMMARY")
        print("="*60)
        
        # 角速度统计
        print("\nAngular Velocity Statistics:")
        for joint_name, ang_vel in ankle_angular_vel.items():
            mean_vel = np.mean(np.abs(ang_vel))
            max_vel = np.max(np.abs(ang_vel))
            std_vel = np.std(ang_vel)
            print(f"  {joint_name}:")
            print(f"    Mean absolute: {np.rad2deg(mean_vel):.2f}°/s")
            print(f"    Maximum absolute: {np.rad2deg(max_vel):.2f}°/s")
            print(f"    Standard deviation: {np.rad2deg(std_vel):.2f}°/s")
        
        # 线速度统计
        print("\nLinear Velocity Statistics:")
        for body_name, lin_vel_data in ankle_linear_vel.items():
            vel_mag = lin_vel_data['velocity_magnitude']
            mean_vel = np.mean(vel_mag)
            max_vel = np.max(vel_mag)
            std_vel = np.std(vel_mag)
            print(f"  {body_name}:")
            print(f"    Mean magnitude: {mean_vel*1000:.2f} mm/s")
            print(f"    Maximum magnitude: {max_vel*1000:.2f} mm/s")
            print(f"    Standard deviation: {std_vel*1000:.2f} mm/s")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ankle joint velocities from MOT files')
    parser.add_argument('--mot_path', type=str, default='results/3_subresults.mot',
                       help='Path to the MOT file to analyze')
    parser.add_argument('--output_dir', type=str, default='ankle_analysis_output',
                       help='Directory to save analysis results')
    parser.add_argument('--max_print_frames', type=int, default=20,
                       help='Maximum number of frames to print in detailed analysis')
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create velocity plots')
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 创建分析器
        print("Initializing ankle motion analyzer...")
        analyzer = AnkleMotionAnalyzer(args.mot_path)
        
        # 计算角速度
        print("\nComputing ankle angular velocities...")
        ankle_angular_vel, time_velocity = analyzer.compute_ankle_angular_velocities()
        
        # 计算线速度
        print("\nComputing ankle linear velocities...")
        ankle_linear_vel, time_velocity = analyzer.compute_ankle_linear_velocities()
        
        # 打印逐帧分析
        print("\nPrinting frame-by-frame analysis...")
        analyzer.print_frame_by_frame_analysis(
            ankle_angular_vel, ankle_linear_vel, time_velocity, 
            max_frames=args.max_print_frames
        )
        
        # 打印统计摘要
        analyzer.print_statistics_summary(ankle_angular_vel, ankle_linear_vel)
        
        # 保存详细结果
        print("\nSaving detailed results...")
        analyzer.save_detailed_results(
            ankle_angular_vel, ankle_linear_vel, time_velocity, args.output_dir
        )
        
        # 创建图表
        if args.create_plots:
            print("\nCreating velocity plots...")
            analyzer.create_velocity_plots(
                ankle_angular_vel, ankle_linear_vel, time_velocity, args.output_dir
            )
        
        print("\nAnkle motion analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()