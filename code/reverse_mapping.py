import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import sys

UPPER_PLANE_POINTS = [1, 3, 7]
LOWER_PLANE_POINTS = [3, 4, 7]
STROKE_ORDER_OF_FIT = 3
DEVIATION_ORDER_OF_FIT = 7
WING_PITCH_ORDER_OF_FIT = 7
N_PCA_COMPONENTS = 25


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
            self.wing_pitch_angle_left = f['wings_psi_left'][:] # shape (num_frames,)
            self.wing_pitch_angle_right = f['wings_psi_right'][:] # shape (num_frames,)
            self.variable_config = {
                'stroke': STROKE_ORDER_OF_FIT,
                'deviation': DEVIATION_ORDER_OF_FIT,
                'wing_pitch': WING_PITCH_ORDER_OF_FIT
            }
            self.right_peak_indices = extract_and_validate_peaks(self.stroke_angle_right)
            self.left_peak_indices = extract_and_validate_peaks(self.stroke_angle_left)

    def set_order_of_fit(self, stroke_order, deviation_order, wing_pitch_order):
        self.variable_config = {
            'stroke': stroke_order,
            'deviation': deviation_order,
            'wing_pitch': wing_pitch_order
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
    
    def calculate_explained_kinematic_variance(self, xl, xp, fpca_xl, fpca_xp, n_comp_xl, n_comp_xp):
        """
        Calculates how much of the raw kinematic variance (time-domain) is explained by:
        1. The Fourier Fit
        2. The PCA Reduced Model
        """
        if not hasattr(self, 'row_map'):
            print("Error: Missing row map.")
            return

        print(f"PCA Config: XL={n_comp_xl}, XP={n_comp_xp}")

        # reconstruct xl and xp from truncated PCs
        w_xl = fpca_xl['w'][:, :n_comp_xl]
        v_xl = fpca_xl['v'][:, :n_comp_xl]
        pca_reconstructed_xl = np.dot(w_xl, v_xl.T) + fpca_xl['mean_vector']

        w_xp = fpca_xp['w'][:, :n_comp_xp]
        v_xp = fpca_xp['v'][:, :n_comp_xp]
        pca_reconstructed_xp = np.dot(w_xp, v_xp.T) + fpca_xp['mean_vector']

        stats = {key: {'Raw_Variance': 0.0, 'SSE_fourier': 0.0, 'SSE_pca': 0.0} for key in self.variable_config}

        for row_idx, meta in enumerate(self.row_map):
            side = meta['side']
            start = meta['start_idx']
            end = meta['end_idx']
            
            xp_pointer = 0
            
            for i, angle_name in enumerate(self.variable_config.keys()):
                order = self.variable_config[angle_name]
                
                raw_signal = getattr(self, f"{angle_name}_angle_{side}")[start:end]
                num_points = len(raw_signal)

                lin_c = xl[row_idx, i]
                num_periodic = 1 + 2 * order
                per_c = xp[row_idx, xp_pointer : xp_pointer + num_periodic]
                fourier_reconstructed_signal = reconstruct_signal_from_fourier_coefs(lin_c, per_c, num_points, order)

                lin_c_pca = pca_reconstructed_xl[row_idx, i]
                per_c_pca = pca_reconstructed_xp[row_idx, xp_pointer : xp_pointer + num_periodic]
                pca_reconstructed_signal = reconstruct_signal_from_fourier_coefs(lin_c_pca, per_c_pca, num_points, order)

                xp_pointer += num_periodic

                stats[angle_name]['Raw_Variance'] += np.sum((raw_signal - np.mean(raw_signal))**2)
                stats[angle_name]['SSE_fourier'] += np.sum((raw_signal - fourier_reconstructed_signal)**2)
                stats[angle_name]['SSE_pca'] += np.sum((raw_signal - pca_reconstructed_signal)**2)

        # print headers
        print(f"{'Variable':<12} | {'Fourier Expl. Var':<18} | {'PCA Expl. Var':<18} | {'Loss due to PCA':<15}")
        print("-" * 70)
        
        for angle_name in self.variable_config:
            sst = stats[angle_name]['Raw_Variance']
            sse_fourier = stats[angle_name]['SSE_fourier']
            sse_pca = stats[angle_name]['SSE_pca']

            # calculate R² values
            r2_fourier = 1.0 - (sse_fourier / sst)
            r2_pca = 1.0 - (sse_pca / sst)
            loss = r2_fourier - r2_pca

            print(f"{angle_name:<12} | {r2_fourier*100:6.2f}%            | {r2_pca*100:6.2f}%            | {loss*100:6.2f}%")

        print("-" * 70)
    
    def verify_fourier_reconstruction(self, xl, xp, row_idx_to_check=0):
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
        angle_names = list(self.variable_config.keys()) # ['stroke', 'deviation', 'wing_pitch']
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
            
            reconstructed_signal = reconstruct_signal_from_fourier_coefs(lin_c, per_c, num_points, order)
            
            ax = axes[i]
            ax.plot(original_signal, 'bo',  linewidth=2, label='Original (Raw)')
            ax.plot(reconstructed_signal, 'r--', linewidth=2, label='Reconstructed (Fit)')
            
            rmse = np.sqrt(np.mean((original_signal - reconstructed_signal)**2))
            std_dev = np.std(original_signal)
            if std_dev > 1e-9:
                nrmse = rmse / std_dev
            else:
                nrmse = 0.0
            ax.set_title(f"{angle_name.capitalize()} Angle (Order {order}) - NRMSE: {nrmse:.4%}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Angle (rad)")

        axes[-1].set_xlabel("Frame Index (within beat)")
        plt.tight_layout()
        if 'OUTPUT_DIR' in globals():
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

    @staticmethod
    def perform_fpca(data_matrix):
        """
        performs FPCA on the given data matrix using SVD.
        """
        mean_vec = np.mean(data_matrix, axis=0)
        centered_data = data_matrix - mean_vec
        u, s_vals, vt = np.linalg.svd(centered_data, full_matrices=False)
        l = np.diag(s_vals)
        w = np.dot(u, l)
        v = vt.T

        explained_variance = (s_vals ** 2) / (data_matrix.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance

        return {
            'mean_vector': mean_vec,
            'u': u,
            'l': l,
            's_vals': s_vals,
            'w': w,
            'v': v,
            'explained_variance_ratio': explained_variance_ratio
        }
    
    @staticmethod
    def analyze_couplings(xl, xp):
        """
        applies FPCA to xl and xp matrices
        """
        print("running fpca")
        fpca_xl = FlightData.perform_fpca(xl)
        # print(f"XL (Linear): Input shape {xl.shape}")
        # print(f"   Top 3 PCs explain: {np.sum(fpca_xl['explained_variance_ratio'][:3])*100:.2f}% variance")

        fpca_xp = FlightData.perform_fpca(xp)
        print(f"XP (Periodic): Input shape {xp.shape}")
        print(f"   Top {N_PCA_COMPONENTS} PCs explain: {np.sum(fpca_xp['explained_variance_ratio'][:N_PCA_COMPONENTS])*100:.2f}% variance")

        return fpca_xl, fpca_xp
    
    def reconstruct_xp_and_calculate_error(self, xp, fpca_dict, n_components=N_PCA_COMPONENTS):
        w_tilde = fpca_dict['w'][:, :n_components]
        v_tilde = fpca_dict['v'][:, :n_components]
        mean_vec = fpca_dict['mean_vector']
        xp_recon = np.dot(w_tilde, v_tilde.T) + mean_vec
        error_matrix = (xp - xp_recon)**2
        mse = np.mean(error_matrix)

        stroke_cols = 1 + 2 * self.variable_config['stroke']
        error_stroke = np.mean(error_matrix[:, :stroke_cols])

        deviation_cols = 1 + 2 * self.variable_config['deviation']
        start_dev = stroke_cols
        end_dev = start_dev + deviation_cols
        error_deviation = np.mean(error_matrix[:, start_dev:end_dev])

        start_wing_pitch = end_dev
        error_wing_pitch = np.mean(error_matrix[:, start_wing_pitch:])

        print(f'--- reconstruction errors with {n_components} PCs ---')
        print(f'Total MSE: {mse:.6f}')
        print(f'Stroke MSE: {error_stroke:.6f}')
        print(f'Deviation MSE: {error_deviation:.6f}')
        print(f'Wing_pitch MSE: {error_wing_pitch:.6f}')

        return xp_recon

    def verify_full_pipeline_reconstruction(
            self,
            fpca_xl, fpca_xp,
            row_idx_to_check=0,
            n_components_xl=3,
            n_components_xp=N_PCA_COMPONENTS):
        
        if not hasattr(self, 'row_map'):
            print("Error: Please run construct_coefficient_matrices first to generate row mapping.")
            return
        
        meta = self.row_map[row_idx_to_check]
        side = meta['side']
        start = meta['start_idx']
        end = meta['end_idx']
        beat_idx = meta['beat_index']

        print(f"Verifying full pipeline reconstruction for beat {beat_idx} ({side}), Frames {start}-{end}")
        print(f"Using {n_components_xl} PCs for xl and {n_components_xp} PCs for xp")

        w_xl = fpca_xl['w'][row_idx_to_check, :n_components_xl]
        v_xl = fpca_xl['v'][:, :n_components_xl]
        mean_xl = fpca_xl['mean_vector']
        xl_recon = np.dot(w_xl, v_xl.T) + mean_xl

        w_xp = fpca_xp['w'][row_idx_to_check, :n_components_xp]
        v_xp = fpca_xp['v'][:, :n_components_xp]
        mean_xp = fpca_xp['mean_vector']
        xp_recon = np.dot(w_xp, v_xp.T) + mean_xp

        w_xl_full = fpca_xl['w'][row_idx_to_check, :]
        v_xl_full = fpca_xl['v'][:, :]
        xl_full_recon = np.dot(w_xl_full, v_xl_full.T) + mean_xl

        w_xp_full = fpca_xp['w'][row_idx_to_check, :]
        v_xp_full = fpca_xp['v'][:, :]
        xp_full = np.dot(w_xp_full, v_xp_full.T) + mean_xp

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        xp_pointer = 0 # Pointer to track position in the flattened xp row
        angle_names = list(self.variable_config.keys()) # ['stroke', 'deviation', 'wing_pitch']

        for i, angle_name in enumerate(angle_names):
            order = self.variable_config[angle_name]
            angle_array = getattr(self, f"{angle_name}_angle_{side}")
            original_signal = angle_array[start:end]
            num_points = len(original_signal)
            
            lin_c_trunc = xl_recon[i]
            num_periodic = 1 + 2 * order
            per_c_trunc = xp_recon[xp_pointer : xp_pointer + num_periodic]

            lin_c_full = xl_full_recon[i]
            per_c_full = xp_full[xp_pointer : xp_pointer + num_periodic]

            xp_pointer += num_periodic

            reconstructed_signal_trunc = reconstruct_signal_from_fourier_coefs(lin_c_trunc, per_c_trunc, num_points, order)
            reconstructed_signal_full = reconstruct_signal_from_fourier_coefs(lin_c_full, per_c_full, num_points, order)

            rmse_trunc = np.sqrt(np.mean((original_signal - reconstructed_signal_trunc)**2))
            rmse_full = np.sqrt(np.mean((original_signal - reconstructed_signal_full)**2))

            std_dev = np.std(original_signal)
            if std_dev > 1e-9:
                nrmse_trunc = rmse_trunc / std_dev
                nrmse_full = rmse_full / std_dev
            else:
                nrmse_trunc = 0.0
                nrmse_full = 0.0

            ax = axes[i]
            ax.plot(original_signal, 'ko', markersize=4, alpha=0.4, label='Original (Raw)')
            ax.plot(reconstructed_signal_full, 'b-', linewidth=1.5, alpha=0.6, label='Full Reconstruction (All PCs)')
            ax.plot(reconstructed_signal_trunc, 'r--', linewidth=2, label=f'Reduced (PCs: {n_components_xp})')
            ax.set_title(f"{angle_name.capitalize()} Angle NRMSE - Reduced: {nrmse_trunc:.4%}, Full: {nrmse_full:.4%}")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Angle (rad)")
        
        axes[-1].set_xlabel("Frame Index (within beat)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if 'OUTPUT_DIR' in globals():
            plt.savefig(os.path.join(OUTPUT_DIR, f"full_pipeline_reconstruction_beat{beat_idx}_{side}_row{row_idx_to_check}.png"))
            plt.close()

def reconstruct_signal_from_fourier_coefs(linear_coeff, periodic_coeffs, num_points, order):
    """
    Reconstructs the time-domain signal from coefficients.
    Mathematically: Signal = A * coeffs
    """
    phase = np.linspace(0, 2 * np.pi, num_points)
    num_cols = 2 + 2 * order
    design_matrix = create_design_matrix(num_points, num_cols, phase, order)
    full_coeffs = np.concatenate(([linear_coeff], periodic_coeffs))
    reconstructed_signal = design_matrix @ full_coeffs
    return reconstructed_signal

def full_pipeline_verification(fd: FlightData):
    fd.set_order_of_fit(STROKE_ORDER_OF_FIT, DEVIATION_ORDER_OF_FIT, WING_PITCH_ORDER_OF_FIT)
    xl, xp = fd.construct_coefficient_matrices()
    fpca_xl, fpca_xp = FlightData.analyze_couplings(xl, xp)

    fd.calculate_explained_kinematic_variance(
        xl, xp, fpca_xl, fpca_xp, 
        n_comp_xl=3, n_comp_xp=N_PCA_COMPONENTS
    )

    random_row_idx = np.random.randint(0, xl.shape[0])
    print(f"\n--- Running Full Pipeline Verification on Row {random_row_idx} ---")
    fd.verify_full_pipeline_reconstruction(
        fpca_xl, fpca_xp,
        row_idx_to_check=random_row_idx,
        n_components_xl=3,
        n_components_xp=N_PCA_COMPONENTS
    )

def run_fpca_verification(fd: FlightData):
    xl, xp = fd.construct_coefficient_matrices()
    fpca_xl, fpca_xp = FlightData.analyze_couplings(xl, xp)
    xp_recon = fd.reconstruct_xp_and_calculate_error(xp, fpca_xp, n_components=N_PCA_COMPONENTS)

def fourier_verification(fd: FlightData, order_range = ((3, 4), (7, 8), (7, 8)), row_range = (0, 2)):
    ranges = create_ranges(order_range)
    for stroke_order, deviation_order, wing_pitch_order in ranges:
        print(f"\n--- Verifying Orders of Fit: Stroke {stroke_order}, Deviation {deviation_order}, Wing_pitch {wing_pitch_order} ---")
        fd.set_order_of_fit(stroke_order, deviation_order, wing_pitch_order)
        xl, xp = fd.construct_coefficient_matrices()
        for row_idx in range(row_range[0], row_range[1]+1):
            print(f"\n--- Verifying Row Index: {row_idx} ---")
            fd.verify_fourier_reconstruction(xl, xp, row_idx_to_check=row_idx)
    print("Verification completed.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python reverse_mapping.py <flight_data_file_path>")
        sys.exit(1)
    flight_data_file_path = sys.argv[1]
    fd = FlightData(flight_data_file_path)
    # fourier_verification(fd, order_range = ((3, 5), (6, 9), (6, 9)), row_range = (0, 5))
    full_pipeline_verification(fd)

if __name__ == "__main__":
    main()