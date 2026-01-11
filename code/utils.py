import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUTPUT_DIR = os.path.abspath('./data/plots/')

def preprocess(X, permute=(0, 3, 2, 1)):

    # Add singleton dim for single train_images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    if permute != None:
        X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8" or np.max(X) > 1:
        X = X.astype("float32") / 255

    return X

def angle_between_planes(leading_plane_points, trailing_plane_points, reference_point):

    print("Leading Plane Points:\n", leading_plane_points)
    print("Trailing Plane Points:\n", trailing_plane_points)
    print("Reference Point:\n", reference_point)

    upper_set = set([tuple(pt) for pt in leading_plane_points])
    lower_set = set([tuple(pt) for pt in trailing_plane_points])
    shared_tuples = list(upper_set.intersection(lower_set))
    shared_points = np.array(shared_tuples)
    if len(shared_points) != 2:
        raise ValueError("Expected exactly two shared points between planes.")
    
    dists = np.linalg.norm(shared_points - reference_point, axis=1)

    if dists[0] < dists[1]:
        h_start, h_end = shared_points[0], shared_points[1]
    else:
        h_start, h_end = shared_points[1], shared_points[0]

    hinge_vec = h_end - h_start
    # hinge_vec /= np.linalg.norm(hinge_vec)
    
    def get_unique_point(plane_points, hinge_a, hinge_b):
        for pt in plane_points:
            # If point is NOT hinge A AND NOT hinge B, it's the tip
            if (np.linalg.norm(pt - hinge_a) > 1e-6) and \
               (np.linalg.norm(pt - hinge_b) > 1e-6):
                return pt
        raise ValueError("Could not find a unique tip point in the plane.")

    leading_edge_point = get_unique_point(leading_plane_points, h_start, h_end)
    trailing_edge_point = get_unique_point(trailing_plane_points, h_start, h_end)

    # --- 3. Construct Aligned Normals ---
    vec_up = leading_edge_point - h_start
    vec_down = trailing_edge_point - h_start
    
    # Calculate Normals (Cross Product)
    n_up = np.cross(hinge_vec, vec_up)
    # n_up /= np.linalg.norm(n_up)
    
    n_down = np.cross(hinge_vec, vec_down)
    # n_down /= np.linalg.norm(n_down)
    
    # Flip one normal so that 0 degrees = Flat Wing
    n_down = -n_down

    # --- 4. Calculate Signed Angle ---
    cross_normals = np.cross(n_down, n_up)
    sin_angle = np.dot(cross_normals, hinge_vec)
    sign = 1
    if sin_angle < 0:
        sign = -1

    angle_value = sign * np.arccos(np.dot(n_down, n_up)/(np.linalg.norm(n_down)*np.linalg.norm(n_up)))

    # cos_angle = np.dot(n_down, n_up)
    # angle_rad = np.arctan2(sin_angle, cos_angle)
    # angle_value = np.degrees(angle_rad)
    
    return angle_value.astype(float)

def extract_and_validate_peaks(angle_array):
    """
    Extracts indices of local maxima and minima from the phi array 
    and plots them for validation.
    """
    save_path = os.path.join(OUTPUT_DIR, 'phi_peak_validation.png')

    angle_array = np.array(angle_array)
    max_indices, _ = find_peaks(angle_array, distance=45)
    min_indices, _ = find_peaks(-angle_array, distance=45)
            
    plt.figure(figsize=(12, 5))
    plt.plot(angle_array, label='Phi Angle', color='black', linewidth=1.5, alpha=0.7)
    plt.scatter(max_indices, angle_array[max_indices], color='red', s=50, zorder=5, label='Frontstroke Peaks (Max)')
    plt.scatter(min_indices, angle_array[min_indices], color='blue', s=50, zorder=5, label='Backstroke Peaks (Min)')
    
    plt.title("Validation: Phi Angle Peak Extraction")
    plt.xlabel("Frame Index")
    plt.ylabel("Phi Angle (rad)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)

    return min_indices

def test_fourier_implementation():
    # --- 1. Setup the "Ground Truth" ---
    # We will construct a signal using specific coefficients.
    # If the code works, it should recover these EXACT numbers.
    
    # Let's say we want:
    # Linear Slope (Coeff 0):  0.5
    # DC Offset    (Coeff 1):  10.0
    # 1st Harmonic (Coeff 2,3): 2.0 * cos(t),  0.0 * sin(t)
    # 2nd Harmonic (Coeff 4,5): 0.0 * cos(2t), -1.5 * sin(2t)
    # ... others 0
    
    expected_coeffs = np.zeros(12) # For Order 5 (12 total coeffs)
    expected_coeffs[0] = 0.5   # Linear term
    expected_coeffs[1] = 10.0  # DC term
    expected_coeffs[2] = 2.0   # Cos(1t)
    expected_coeffs[5] = -1.5  # Sin(2t)
    
    # --- 2. Generate the Synthetic Signal ---
    num_points = 101 # Same as the paper's "101 evenly-spaced points"
    phase = np.linspace(0, 2*np.pi, num_points)
    
    # Construct y = 0.5*(t-pi) + 10 + 2cos(t) - 1.5sin(2t)
    # Note the paper uses (t - pi) for the linear term!
    linear_term = expected_coeffs[0] * (phase - np.pi) 
    dc_term = expected_coeffs[1]
    harmonics = (expected_coeffs[2] * np.cos(1 * phase) + 
                 expected_coeffs[5] * np.sin(2 * phase))
                 
    signal = linear_term + dc_term + harmonics
    
    # --- 3. Run Your Fitting Function ---
    # (Copying the logic discussed previously)
    def fit_series_paper_style(signal, order=5):
        phase = np.linspace(0, 2*np.pi, len(signal))
        num_cols = 2 + 2 * order
        A = np.zeros((len(signal), num_cols))
        A[:, 0] = phase - np.pi  # Linear
        A[:, 1] = 1.0            # DC
        for k in range(1, order + 1):
            A[:, 2*k]     = np.cos(k * phase)
            A[:, 2*k + 1] = np.sin(k * phase)
        coeffs, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
        return coeffs, A @ coeffs

    calculated_coeffs, fitted_curve = fit_series_paper_style(signal)
    
    # --- 4. Verify Results ---
    print(f"{'Term':<15} | {'Expected':<10} | {'Calculated':<10} | {'Error'}")
    print("-" * 50)
    
    names = ['Linear', 'DC', 'Cos(1t)', 'Sin(1t)', 'Cos(2t)', 'Sin(2t)']
    for i in range(6): # Check first few terms
        err = abs(expected_coeffs[i] - calculated_coeffs[i])
        status = "✅" if err < 1e-10 else "❌"
        print(f"{names[i]:<15} | {expected_coeffs[i]:<10.2f} | {calculated_coeffs[i]:<10.2f} | {status}")

    # --- 5. Visual Check ---
    save_path = os.path.join(OUTPUT_DIR, 'fourier_fit_validation.png')
    plt.figure(figsize=(10, 5))
    plt.plot(phase, signal, 'k.', label='Input Data (Synthetic)', markersize=10)
    plt.plot(phase, fitted_curve, 'r-', label='Fitted Result', linewidth=2)
    plt.title("Validation: Does the Fit Match the Data?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)