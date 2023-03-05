import numpy as np

def filloutliers(data, window_size, n_sigmas=3):
  n = len(data)
  new_data = data.copy()

  # Scale factor for Gaussian distribution
  K = 1.4826

  prev = np.array([np.nan for _ in range(data.shape[1])])
  half_window_size = window_size // 2

  start = -half_window_size
  end = half_window_size + 1

  for i in range(n):
    real_start = max(0, start)
    real_end = min(n, end)

    window = data[real_start : real_end]

    x0 = np.nanmedian(window, axis=0)
    S0 = K * np.nanmedian(np.abs(window - x0), axis=0)

    mad = np.abs(data[i] - x0)

    indexes = np.argwhere(mad > (n_sigmas * S0))
    new_data[i, indexes] = prev[indexes]

    # Update prev values
    not_indexes = np.argwhere(mad <= (n_sigmas * S0))
    prev[not_indexes] = new_data[i, not_indexes]

    start += 1
    end += 1

  return new_data

def single_filloutliers(input_series, window_size, n_sigmas=3):
  n = len(input_series)
  new_series = input_series.copy()
  K = 1.4826 # scale factor for Gaussian distribution
  
  prev = np.nan
  half_window_size = window_size // 2

  start = -half_window_size
  end = half_window_size + 1

  for i in range(n):
    real_start = max(0, start)
    real_end = min(n, end)

    window = input_series[real_start : real_end]
    x0 = np.nanmedian(window)
    S0 = K * np.nanmedian(np.abs(window - x0))

    mad = np.abs(input_series[i] - x0)

    if mad > (n_sigmas * S0):
      new_series[i] = prev
    else:
      prev = new_series[i]

    start += 1
    end += 1
    
  return new_series

def fillmissing(data):
  new_data = data.copy()

  rows, cols = data.shape
  indexes = np.argwhere(np.isnan(data))
    
  for x, y in indexes:

    # Check nearest value
    i = x - 1
    j = x + 1
    
    flag = False
    while True:
      if i > 0:
        element_at_i = data[i, y]
        if not np.isnan(element_at_i):
          new_data[x, y] = element_at_i
          break
        i -= 1
        flag = True
      
      if j < rows:
        element_at_j = data[j, y]
        if not np.isnan(element_at_j):
          new_data[x, y] = element_at_j
          break
        j += 1
        flag = True
      
      if not flag:
        raise Exception(f"Could not find any value near the missing value at index ({x}, {y})")
    
  return new_data

def fillmissing_single(data):
  new_data = data.copy()

  rows = len(data)
  indexes = np.argwhere(np.isnan(data))
    
  for index in indexes:
    # Check nearest value
    i = index - 1
    j = index + 1
    
    flag = False
    while True:
      if i > 0:
        element_at_i = data[i]
        if not np.isnan(element_at_i):
          new_data[index] = element_at_i
          break
        i -= 1
        flag = True
      
      if j < rows:
        element_at_j = data[j]
        if not np.isnan(element_at_j):
          new_data[index] = element_at_j
          break
        j += 1
        flag = True
      
      if not flag:
        raise Exception(f"Could not find any value near the missing value at index {index}")
    
  return new_data

def sanitize_amplitude(sample, window_size=5):
  csi_abs = np.absolute(sample)

  num_packets, subcarriers = csi_abs.shape

  if num_packets == 1:
    return csi_abs
  
  return filloutliers(csi_abs, window_size=window_size)

def sanitize_phase(sample):
  input_phase = np.angle(sample)
  output_phase = np.zeros_like(sample, dtype=np.float64)
  num_packets, subcarriers = sample.shape

  for packet in range(num_packets):
    x = np.unwrap(input_phase[packet, :])
    k = (x[subcarriers-1] - x[0]) / (subcarriers/2 - (- subcarriers/2))
    b = sum(x) / subcarriers

    _range = np.hstack((np.arange(-subcarriers//2, -1 + 1), np.arange(1, subcarriers//2 + 1)))
    output_phase[packet, :] = x - k * _range - b

  return output_phase

def preprocess_sample(data, num_packets, ntx, nrx, subcarriers):
  # Remove guard band NULL subcarries
  raw_csi = data[:num_packets, 12 : -12]
  # Remove LO Leakage subcarriers
  raw_csi = np.delete(raw_csi, [52, 53], axis=1)

  complex_csi = np.zeros((num_packets, subcarriers), dtype=complex)

  # Reformatting raw CSI re-organizing real and imaginary parts
  for packet in range(num_packets):
    index = 0
    for subcarrier in range(subcarriers):
      imag = raw_csi[packet, index]
      real = raw_csi[packet, index + 1]
      complex_csi[packet, subcarrier] = complex(real, imag)
      index += 2

  csi_mat = complex_csi.copy().reshape((num_packets, ntx, nrx, subcarriers))

  final_amplitude = np.zeros((num_packets, ntx, nrx, subcarriers))
  final_phase = np.zeros((num_packets, ntx, nrx, subcarriers))

  # Analyze all TX/RX antenna pairs
  for tx in range(ntx):
    for rx in range(nrx):
      sample = csi_mat[:, tx, rx, :]
      sanitized_amplitude = sanitize_amplitude(sample)
      sanitized_phase = sanitize_phase(sample)

      final_amplitude[:, tx, rx, :] = sanitized_amplitude
      final_phase[:, tx, rx, :] = sanitized_phase

  # Compute median amplitudes
  median_final_amplitude = np.median(final_amplitude, [1, 2])
  median_final_amplitude = median_final_amplitude[np.any(median_final_amplitude, 1), :]
  
  filtered_median_amplitude = filloutliers(median_final_amplitude, window_size=5)

  return csi_mat, final_amplitude, final_phase, median_final_amplitude, filtered_median_amplitude