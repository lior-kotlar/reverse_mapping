import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *

flight_data_file_path = '../pose-estimation-torch/predict_output/debug_outputs/sagiv_free_flight/mov1/mov1_analysis_smoothed.h5' 

UPPER_PLANE_POINTS = [1, 3, 7]
LOWER_PLANE_POINTS = [3, 4, 7]
STROKE_ORDER_OF_FIT = 3
DEVIATION_ORDER_OF_FIT = 7
TWIST_ORDER_OF_FIT = 7


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
            self.variable_config = {
                'stroke': STROKE_ORDER_OF_FIT,
                'deviation': DEVIATION_ORDER_OF_FIT,
                'twist': TWIST_ORDER_OF_FIT
            }
            self.right_peak_indices = extract_and_validate_peaks(self.stroke_angle_right)
            self.left_peak_indices = extract_and_validate_peaks(self.stroke_angle_left)

    def set_order_of_fit(self, order):
        self.variable_config = {
            'stroke': order,
            'deviation': order,
            'twist': order
        }
    
    def construct_coefficient_matrices(self):
        xl_ncols = len(self.variable_config.keys())
        xp_ncols = sum([1 + 2*order for order in self.variable_config.values()])
        rows_list_xl = []
        rows_list_xp = []

        self.row_map = []  # To keep track of which row corresponds to which beat and side

        num_beats = min(len(self.left_peak_indices), len(self.right_peak_indices)) - 1
        for i in range(num_beats):
            for side, peaks in [('right', self.right_peak_indices), ('left', self.left_peak_indices)]:
                start_idx = peaks[i]
                end_idx = peaks[i+1]

                self.row_map.append({
                    'beat_index': i,
                    'side': side,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })

                row_xl = []
                row_xp = []
                for angle, order in self.variable_config.items():
                    angle_array = getattr(self, f"{angle}_angle_{side}")
                    
                    coeffs = coefficients_for_wingbeat(angle_array[start_idx:end_idx], order=order)
                    row_xl.append(coeffs[0])  # Linear Term
                    row_xp.extend(coeffs[1:])  # DC + Harmonics
                
                rows_list_xl.append(row_xl)
                rows_list_xp.append(row_xp)
        
        xl = np.array(rows_list_xl)
        xp = np.array(rows_list_xp)
        # print(f"Coefficient Matrix xl Shape: {xl.shape}, expected ({num_beats*2}, {xl_ncols})")
        # print(f"Coefficient Matrix xp Shape: {xp.shape}, expected ({num_beats*2}, {xp_ncols})")
        return xl, xp
    
    def verify_reconstruction(self, xl, xp, row_idx_to_check=0):
        """
        Picks a row from the coefficient matrices, reconstructs the signal,
        and plots it against the original raw data.
        """
        if not hasattr(self, 'row_map'):
            print("Error: Please run construct_coefficient_matrices first to generate row mapping.")
            return

        meta = self.row_map[row_idx_to_check]
        beat_idx = meta['beat_index']
        side = meta['side']
        start = meta['start_idx']
        end = meta['end_idx']
        
        print(f"Verifying Row {row_idx_to_check}: Beat {beat_idx} ({side}), Frames {start}-{end}")

        row_xl_vals = xl[row_idx_to_check]
        row_xp_vals = xp[row_idx_to_check]
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        angle_names = list(self.variable_config.keys()) # ['stroke', 'deviation', 'twist']
        xp_pointer = 0 # Pointer to track position in the flattened xp row
        
        for i, angle_name in enumerate(angle_names):
            order = self.variable_config[angle_name]
            angle_array = getattr(self, f"{angle_name}_angle_{side}")
            original_signal = angle_array[start:end]
            num_points = len(original_signal)
            
            lin_c = row_xl_vals[i]
            
            num_periodic = 1 + 2 * order # DC + 2*order harmonics
            per_c = row_xp_vals[xp_pointer : xp_pointer + num_periodic]
            xp_pointer += num_periodic # Advance pointer for next angle
            
            reconstructed_signal = reconstruct_wingbeat(lin_c, per_c, num_points, order)
            
            ax = axes[i]
            ax.plot(original_signal, 'b-',  linewidth=2, label='Original (Raw)')
            ax.plot(reconstructed_signal, 'r--', linewidth=2, label='Reconstructed (Fit)')
            
            rmse = np.sqrt(np.mean((original_signal - reconstructed_signal)**2))
            ax.set_title(f"{angle_name.capitalize()} Angle (Order {order}) - RMSE: {rmse:.4f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Angle (rad)")

        axes[-1].set_xlabel("Frame Index (within beat)")
        plt.tight_layout()
        order_string = "_".join([f"{angle}{self.variable_config[angle]}" for angle in angle_names])
        plt.savefig(os.path.join(OUTPUT_DIR, f"reconstruction_verification_row{row_idx_to_check}_order{order_string}.png"))
        plt.close()

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

def verification_loop(fd: FlightData, order_range = (3, 8), row_range = (0, 2)):
    for order in range(order_range[0], order_range[1]+1):
        print(f"\n--- Verifying Order of Fit: {order} ---")
        fd.set_order_of_fit(order)
        xl, xp = fd.construct_coefficient_matrices()
        for row_idx in range(row_range[0], row_range[1]+1):
            print(f"\n--- Verifying Row Index: {row_idx} ---")
            fd.verify_reconstruction(xl, xp, row_idx_to_check=row_idx)
    print("Verification completed.")

def main():
    fd = FlightData(flight_data_file_path)


if __name__ == "__main__":
    main()