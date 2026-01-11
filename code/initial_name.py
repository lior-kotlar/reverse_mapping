import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import angle_between_planes, extract_and_validate_peaks, OUTPUT_DIR

flight_data_file_path = '../pose-estimation-torch/predict_output/debug_outputs/sagiv_free_flight/mov1/mov1_analysis_smoothed.h5' 

UPPER_PLANE_POINTS = [1, 3, 7]
LOWER_PLANE_POINTS = [3, 4, 7]


class FlightData:
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            # right_wing_inds = f['right_inds'][:] # shape (num_points_per_wing,)
            # left_wing_inds = f['left_inds'][:] # shape (num_points_per_wing,)
            # all_points_3d = f['points_3D'][:] # shape (num_frames, num_points=18, 3)
            # right_wing_points = all_points_3d[:, right_wing_inds, :] # shape (num_frames, num_points_per_wing, 3)
            # left_wing_points = all_points_3d[:, left_wing_inds, :] # shape (num_frames, num_points_per_wing, 3)
            # self.right_wing_leading_plane_points = right_wing_points[:, UPPER_PLANE_POINTS, :] # shape (num_frames, 3, 3)
            # self.right_wing_trailing_plane_points = right_wing_points[:, LOWER_PLANE_POINTS, :] # shape (num_frames, 3, 3)
            # self.left_wing_leading_plane_points = left_wing_points[:, UPPER_PLANE_POINTS, :] # shape (num_frames, 3, 3)
            # self.left_wing_trailing_plane_points = left_wing_points[:, LOWER_PLANE_POINTS, :] # shape (num_frames, 3, 3)
            self.stroke_angle_left = f['wings_phi_left'][:] # shape (num_frames,)
            self.stroke_angle_right = f['wings_phi_right'][:] # shape (num_frames,)
            self.deviation_angle_left = f['wings_theta_left'][:] # shape (num_frames,)
            self.deviation_angle_right = f['wings_theta_right'][:] # shape (num_frames,)
            self.twist_angle_left = f['wings_psi_left'][:] # shape (num_frames,)
            self.twist_angle_right = f['wings_psi_right'][:] # shape (num_frames,)
            self.variable_cofig = {
                'stroke': 5,
                'deviation': 5,
                'twist': 5
            }

    def coefficients_for_wingbeat(self, wingbeat_signal, order=5):
        num_points = len(wingbeat_signal)
        
        phase = np.linspace(0, 2 * np.pi, num_points)
        
        num_cols = 2 + 2 * order
        A = np.zeros((num_points, num_cols))
        
        A[:, 0] = phase - np.pi  # Linear Term
        A[:, 1] = 1.0            # DC Term
        
        for k in range(1, order + 1):
            A[:, 2*k]     = np.cos(k * phase)
            A[:, 2*k + 1] = np.sin(k * phase)
            
        coeffs, _, _, _ = np.linalg.lstsq(A, wingbeat_signal, rcond=None)
        return coeffs
    
    def construct_coefficient_matrices(self, peak_indices_right, peak_indices_left):
        xl_ncols = len(self.variable_cofig.keys())
        xp_ncols = sum([1 + 2*order for order in self.variable_cofig.values()])
        rows_list_xl = []
        rows_list_xp = []
        num_beats = min(len(peak_indices_left), len(peak_indices_right)) - 1
        for i in range(num_beats):
            for side, peaks in [('right', peak_indices_right), ('left', peak_indices_left)]:
                start_idx = peaks[i]
                end_idx = peaks[i+1]
                row_xl = []
                row_xp = []
                for angle, order in self.variable_cofig.items():
                    angle_array = getattr(self, f"{angle}_angle_{side}")
                    
                    coeffs = self.coefficients_for_wingbeat(angle_array[start_idx:end_idx], order=order)
                    row_xl.append(coeffs[0])  # Linear Term
                    row_xp.extend(coeffs[1:])  # DC + Harmonics
                
                rows_list_xl.append(row_xl)
                rows_list_xp.append(row_xp)
        
        xl = np.array(rows_list_xl)
        xp = np.array(rows_list_xp)
        print(f"Coefficient Matrix xl Shape: {xl.shape}, expected ({num_beats*2}, {xl_ncols})")
        print(f"Coefficient Matrix xp Shape: {xp.shape}, expected ({num_beats*2}, {xp_ncols})")
        return xl, xp
    
    def from_raw_data_to_coef_matrices(self):
        right_peak_indices = extract_and_validate_peaks(self.stroke_angle_right)
        left_peak_indices = extract_and_validate_peaks(self.stroke_angle_left)
        xl, xp = self.construct_coefficient_matrices(right_peak_indices, left_peak_indices)
        return xl, xp

    def calculate_signed_deformation_angle_per_frame(self, frame):
        """
        Calculates the signed deformation angle, robust to global position.
        leading_plane_points (np.array): shape (3, 3) upper wing points
        trailing_plane_points (np.array): shape (3, 3) lower wing points
        reference_point (np.array): shape (3,) a point known to be closer
                                    to the wing root than the wing tip
        """
        right_leading_plane_points = self.right_wing_leading_plane_points[frame]
        right_trailing_plane_points = self.right_wing_trailing_plane_points[frame]
        left_leading_plane_points = self.left_wing_leading_plane_points[frame]
        left_trailing_plane_points = self.left_wing_trailing_plane_points[frame]
        right_left_deformation_angles = []
        for leading_plane_points, trailing_plane_points, reference_point in [
            (right_leading_plane_points, right_trailing_plane_points, right_leading_plane_points[2]),
            (left_leading_plane_points, left_trailing_plane_points, left_leading_plane_points[2])
        ]:
            angle_deg = angle_between_planes(leading_plane_points, trailing_plane_points, reference_point)
            
            right_left_deformation_angles.append(angle_deg)
        
        return right_left_deformation_angles[0], right_left_deformation_angles[1]

    def deformation_angle_over_time(self):
        num_frames = self.right_wing_leading_plane_points.shape[0]
        right_wing_angles = []
        left_wing_angles = []
        for frame in range(num_frames):
            right_angle, left_angle = self.calculate_signed_deformation_angle_per_frame(frame)
            right_wing_angles.append(right_angle)
            left_wing_angles.append(left_angle)
        return np.array(right_wing_angles), np.array(left_wing_angles)
    
    def plot_deformation_angles(self):
        save_path = os.path.join(OUTPUT_DIR, 'wing_deformation_angles.png')
        right_wing_angles, left_wing_angles = self.deformation_angle_over_time()
        plt.figure(figsize=(10, 5))
        # plt.plot(right_wing_angles, label='Right Wing Deformation Angle')
        plt.plot(left_wing_angles, label='Left Wing Deformation Angle')
        plt.xlabel('Frame')
        plt.ylabel('Deformation Angle (degrees)')
        plt.title('Wing Deformation Angles Over Time')
        plt.legend()
        plt.grid()
        plt.savefig(save_path)

def main():
    fd = FlightData(flight_data_file_path)
    xl, xp = fd.from_raw_data_to_coef_matrices()
    print("Coefficient Matrices Computed.")


if __name__ == "__main__":
    main()