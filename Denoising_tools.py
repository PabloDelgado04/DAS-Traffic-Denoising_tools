import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import solve
from scipy.ndimage import uniform_filter1d
import pywt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_erosion, label, find_objects

# PREPROCESSING TOOLS (Architecture Proposed)

def ricker_manual(width):
    """Generates a normalized Ricker wavelet pulse."""
    t = np.linspace(-2, 2, width)
    pulse = (1 - 2 * (np.pi**2) * (t**2)) * np.exp(-(np.pi**2) * (t**2))
    return pulse / np.max(np.abs(pulse))

def preparar_entrada_lineal(data, mask, factor_amp=0.01):
    """
    Regularizes trajectories by replacing real signatures with Ricker wavelets 
    to create a smooth guide map for the f-x Wiener filter.
    """
    Nt, Nx = data.shape
    data_fake = data.copy()
    data_fake[mask] = 0
    
    labeled_array, num_features = label(mask)
    objs = find_objects(labeled_array)
    pulse = ricker_manual(200)
    half_p = 100
    amp_ref = np.max(np.abs(data)) * factor_amp
    restoration_map = []
    
    for idx, sl in enumerate(objs):
        if sl is None: continue
        mask_island = (labeled_array == (idx + 1))
        restoration_map.append((mask_island, data[mask_island].copy()))
        
        coords = np.argwhere(mask_island)
        m, b = np.polyfit(coords[:, 1] + sl[1].start, coords[:, 0] + sl[0].start, 1)
        
        for s_idx in range(sl[1].start, sl[1].stop):
            t_ideal = int(m * s_idx + b)
            if 0 <= t_ideal < Nt:
                t_i, t_f = max(0, t_ideal - half_p), min(Nt, t_ideal + half_p)
                p_s, p_e = half_p - (t_ideal - t_i), half_p + (t_f - t_ideal)
                data_fake[t_i:t_f, s_idx] = np.maximum(data_fake[t_i:t_f, s_idx], pulse[p_s:p_e] * amp_ref)
                
    return data_fake, restoration_map

def restaurar_firmas(data_filtered, restoration_map):
    """Restores real acoustic signatures into the filtered data using the map."""
    out = data_filtered.copy()
    for mask_island, original_vals in restoration_map:
        out[mask_island] = original_vals
    return out

def detect_mask_separated(data):
    """
    Intelligent event detection: extracts the vehicle trajectory mask 
    using Hilbert transform and robust MAD thresholding.
    """
    Nt, Nx = data.shape
    envelope = np.abs(hilbert(data, axis=0))
    mask = np.zeros_like(data, dtype=bool)
    
    for i in range(Nx):
        trace_smooth = gaussian_filter1d(envelope[:, i], sigma=60)
        mad = np.median(np.abs(trace_smooth - np.median(trace_smooth)))
        thresh = np.median(trace_smooth) + (2.5 * mad * 1.4826)
        mask[:, i] = trace_smooth > thresh
        
    mask_sep = binary_closing(mask, np.ones((20, 1)))
    mask_sep = binary_erosion(mask_sep, structure=np.ones((150, 1)))
    labeled, n = label(mask_sep)
    objs = find_objects(labeled)
    
    clean_mask = np.zeros_like(mask)
    for idx, sl in enumerate(objs):
        if (sl[1].stop - sl[1].start) >= 3:
            clean_mask[sl] |= (labeled[sl] == (idx + 1))
    return clean_mask


# 2. FILTERING TOOLS (f-x Wiener & Wavelet)

def apply_fx_wiener_denoise(noisy_data, p=5, window_size=24, step=4, eps=1e-6, mode='forward-backward', smooth_len=5, agresividad=1.0):
    """
    f-x Wiener filter implementation using Yule-Walker equations.
    Features directional (forward-backward) filtering for edge preservation.
    """
    print(f"Starting f-x Wiener filtering ({mode}). Aggressiveness: {agresividad}.")
    Nt, Nx = noisy_data.shape
    
    # 1. Transform to frequency domain
    Y_fw = fft(noisy_data, axis=0)
    Nf = (Nt // 2) + 1 
    Y_clean_fw = np.zeros_like(Y_fw, dtype=complex) 
    
    # Process positive frequencies
    for k in range(1, Nf):
        Y_z = Y_fw[k, :]  
        Y_clean_k = np.zeros(Nx, dtype=complex)
        weights_k = np.zeros(Nx, dtype=float)

        inicios = list(range(0, max(1, Nx - window_size + 1), step)) 
        if inicios[-1] + window_size < Nx:
            inicios.append(max(0, Nx - window_size))   
            
        # 2. Spatial windowing
        for start in inicios:
            end = min(start + window_size, Nx)
            if end - start <= p: continue      
                
            Y_win = Y_z[start:end]             
            N_valid = len(Y_win) - p            
            
            # --- FORWARD PASS ---
            Z_f = np.zeros((N_valid, p), dtype=complex)      
            Y_t_f = np.zeros(N_valid, dtype=complex)          
            
            for i in range(N_valid):                        
                z_idx = i + p                           
                Z_f[i, :] = Y_win[z_idx-p : z_idx][::-1]   
                Y_t_f[i] = Y_win[z_idx]                    
                
            Z_H_f = Z_f.conj().T                       
            Rzz_f = Z_H_f @ Z_f                        
            rzy_f = Z_H_f @ Y_t_f                      
            
            reg_f = eps * np.trace(np.abs(Rzz_f))
            if reg_f == 0: reg_f = eps
            Rzz_f += np.eye(p) * reg_f
    
            try:
                a_f = solve(Rzz_f, rzy_f)                      
            except np.linalg.LinAlgError:
                a_f = np.zeros(p, dtype=complex)
                
            Y_hat_f = Z_f @ a_f                        
            E_f = Y_t_f - Y_hat_f                      
            
            # --- WIENER GAIN CALCULATION ---
            S_YY_f = uniform_filter1d(np.abs(Y_hat_f)**2, size=smooth_len)
            S_EE_f = uniform_filter1d(np.abs(E_f)**2, size=smooth_len)
            G_f = S_YY_f / (S_YY_f + S_EE_f + eps)
            Y_clean_local_f = Y_hat_f + G_f * E_f                

            if mode == 'forward':
                Y_clean_local = Y_clean_local_f
                start_insert, end_insert, N_insert = start + p, end, N_valid
                
            elif mode == 'forward-backward':
                # --- BACKWARD PASS ---
                Y_win_rev = Y_win[::-1] 
                Z_b = np.zeros((N_valid, p), dtype=complex)      
                Y_t_b = np.zeros(N_valid, dtype=complex)
                
                for i in range(N_valid):                        
                    z_idx = i + p                           
                    Z_b[i, :] = Y_win_rev[z_idx-p : z_idx][::-1]   
                    Y_t_b[i] = Y_win_rev[z_idx]
                
                Z_H_b = Z_b.conj().T                       
                Rzz_b = Z_H_b @ Z_b                        
                rzy_b = Z_H_b @ Y_t_b 
                
                reg_b = eps * np.trace(np.abs(Rzz_b))
                if reg_b == 0: reg_b = eps
                Rzz_b += np.eye(p) * reg_b
        
                try:
                    a_b = solve(Rzz_b, rzy_b)                      
                except np.linalg.LinAlgError:
                    a_b = np.zeros(p, dtype=complex)
                
                Y_hat_b_rev = Z_b @ a_b
                E_b_rev = Y_t_b - Y_hat_b_rev
                
                S_YY_b = uniform_filter1d(np.abs(Y_hat_b_rev)**2, size=smooth_len)
                S_EE_b = uniform_filter1d(np.abs(E_b_rev)**2, size=smooth_len)
                G_b = S_YY_b / (S_YY_b + S_EE_b + eps)
                Y_clean_local_b = (Y_hat_b_rev + G_b * E_b_rev)[::-1]
                
                # --- FUSION LOGIC ---
                N_win = len(Y_win)
                Y_combined, W_combined = np.zeros(N_win, dtype=complex), np.zeros(N_win, dtype=float)
                Y_combined[p:] += Y_clean_local_f
                W_combined[p:] += 1.0
                Y_combined[:-p] += Y_clean_local_b
                W_combined[:-p] += 1.0
                
                mask_w = W_combined > 0
                Y_combined[mask_w] /= W_combined[mask_w]
                Y_clean_local = Y_combined
                start_insert, end_insert, N_insert = start, end, N_win
            
            # Reconstruction (Overlap-Add)
            window_func = np.hanning(N_insert)             
            Y_clean_k[start_insert : end_insert] += Y_clean_local * window_func
            weights_k[start_insert : end_insert] += window_func      
            
        # Normalization
        mask = weights_k > 0
        Y_clean_k[mask] /= weights_k[mask]        
        Y_clean_k[~mask] = Y_z[~mask]
        
        # Aggressiveness parameter applied
        Y_clean_k = (agresividad * Y_clean_k) + ((1.0 - agresividad) * Y_z)
        Y_clean_fw[k, :] = Y_clean_k
        
    Y_clean_fw[0, :] = Y_fw[0, :]                        

    # Symmetry restoration and IFFT
    for k in range(1, Nf):
        if k == Nt / 2: continue
        Y_clean_fw[Nt - k, :] = np.conj(Y_clean_fw[k, :])
        
    return np.real(ifft(Y_clean_fw, axis=0))

def wavelet_denoise(data, wavelet='db4', level=4, factor=2.0):
    """
    Denoising using Discrete Wavelet Transform. 
    Applies soft thresholding based on local MAD estimation per level.
    """
    denoised = np.zeros_like(data)
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data[:, i], wavelet, level=level)
        new_coeffs = [coeffs[0]] 
        
        for j in range(1, len(coeffs)):
            sigma_j = np.median(np.abs(coeffs[j])) / 0.6745
            umbral_j = factor * sigma_j
            filtrado = pywt.threshold(coeffs[j], umbral_j, mode='soft')
            new_coeffs.append(filtrado)
            
        denoised[:, i] = pywt.waverec(new_coeffs, wavelet)[:data.shape[0]]
    return denoised

