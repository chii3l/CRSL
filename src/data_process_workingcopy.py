import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import pywt
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif']=['SimHei'] #鐢ㄦ潵姝ｅ父鏄剧ず涓枃鏍囩
plt.rcParams['axes.unicode_minus']=False #鐢ㄦ潵姝ｅ父鏄剧ず璐熷彿

def _require_columns(df, required_columns, sheet_name):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Sheet '{sheet_name}' is missing columns: {missing_columns}")


def _interp_curve(df, group_column, group_value, value_column, wavelengths, sheet_name, outside_value=0.0):
    curve = df[df[group_column].astype(str) == str(group_value)]
    if curve.empty:
        raise ValueError(f"Sheet '{sheet_name}' has no rows for {group_column}={group_value}")

    curve = curve.sort_values('wavelength_nm')
    x = curve['wavelength_nm'].to_numpy(dtype=float)
    y = curve[value_column].to_numpy(dtype=float)
    return np.interp(wavelengths, x, y, left=outside_value, right=outside_value)


def _normalize_observation_row(row, wavelengths, normalize):
    if normalize is None or normalize == 'none':
        return row
    if normalize == 'sum':
        denom = row.sum()
    elif normalize == 'area':
        denom = np.trapz(row, wavelengths)
    elif normalize == 'max':
        denom = row.max()
    else:
        raise ValueError("normalize must be one of: 'sum', 'area', 'max', 'none', None")

    if denom > 0:
        return row / denom
    return row


def _read_legacy_curve(xls, sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Sheet '{sheet_name}' must contain at least two columns")
    df = df.iloc[:, :2].copy()
    df.columns = ['wavelength_nm', 'value']
    df = df.dropna().sort_values('wavelength_nm')
    return df


def _build_observation_matrix_from_legacy_sheets(
        xls,
        spectral_min,
        spectral_max,
        spectral_step,
        normalize,
        legacy_channel_order,
        use_abs):
    led_sheet_names = ['665', '905', '1200', '1300', '1460', '1550']
    channel_order = legacy_channel_order or ['1200', '1300', '1460', '1550', '665', '905']

    led_curves = {sheet: _read_legacy_curve(xls, sheet) for sheet in led_sheet_names}
    pd_curves = {
        'SI': _read_legacy_curve(xls, 'SI'),
        'InGaAs': _read_legacy_curve(xls, 'InGaAs')
    }
    abs_curve = _read_legacy_curve(xls, 'Abs') if use_abs and 'Abs' in xls.sheet_names else None

    if spectral_min is None:
        spectral_min = min(float(led_curves[sheet]['wavelength_nm'].min()) for sheet in channel_order)
    if spectral_max is None:
        spectral_max = max(float(led_curves[sheet]['wavelength_nm'].max()) for sheet in channel_order)
    if spectral_step <= 0:
        raise ValueError('spectral_step must be positive')

    wavelengths = np.arange(float(spectral_min), float(spectral_max) + spectral_step * 0.5, spectral_step)
    H_rows = []

    for led_name in channel_order:
        if led_name not in led_curves:
            raise ValueError(f"Legacy spectral table has no LED sheet '{led_name}'")

        pd_name = 'SI' if led_name in ['665', '905'] else 'InGaAs'
        led_df = led_curves[led_name]
        pd_df = pd_curves[pd_name]
        led_curve = np.interp(
            wavelengths,
            led_df['wavelength_nm'].to_numpy(dtype=float),
            led_df['value'].to_numpy(dtype=float),
            left=0.0,
            right=0.0)
        pd_curve = np.interp(
            wavelengths,
            pd_df['wavelength_nm'].to_numpy(dtype=float),
            pd_df['value'].to_numpy(dtype=float),
            left=0.0,
            right=0.0)

        prior_curve = np.ones_like(wavelengths, dtype=float)
        if abs_curve is not None:
            absorbance = np.interp(
                wavelengths,
                abs_curve['wavelength_nm'].to_numpy(dtype=float),
                abs_curve['value'].to_numpy(dtype=float),
                left=0.0,
                right=0.0)
            prior_curve = 10 ** (-absorbance)

        wavelength_min = float(led_df['wavelength_nm'].min())
        wavelength_max = float(led_df['wavelength_nm'].max())
        band_mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
        row = led_curve * pd_curve * prior_curve
        row[~band_mask] = 0.0
        row = np.clip(row, 0.0, None)
        H_rows.append(_normalize_observation_row(row, wavelengths, normalize))

    return np.stack(H_rows, axis=0), wavelengths


def build_observation_matrix(
        excel_path,
        spectral_min=None,
        spectral_max=None,
        spectral_step=5.0,
        normalize='sum',
        legacy_channel_order=None,
        use_abs=True):
    """
    Build the LED-PD observation matrix H from spectral response tables.

    Standard sheets:
        led_spectra: led_name, wavelength_nm, intensity
        pd_responsivity: pd_name, wavelength_nm, responsivity
        channel_map: channel, led_name, pd_name, wavelength_min, wavelength_max

    Standard optional sheet:
        optional_prior: prior_name, wavelength_nm, value, value_type

    Legacy sheets are also supported:
        665, 905, 1200, 1300, 1460, 1550, SI, InGaAs, Abs(optional)

    Returns:
        H: array with shape (n_channels, n_wavelengths)
        wavelengths: wavelength grid in nm
    """
    xls = pd.ExcelFile(excel_path)
    required_sheets = ['led_spectra', 'pd_responsivity', 'channel_map']
    legacy_sheets = ['665', '905', '1200', '1300', '1460', '1550', 'SI', 'InGaAs']
    sheet_names = set(xls.sheet_names)
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
    if missing_sheets:
        missing_legacy_sheets = [sheet for sheet in legacy_sheets if sheet not in sheet_names]
        if missing_legacy_sheets:
            xls.close()
            raise ValueError(
                f"Missing required sheets: {missing_sheets}. "
                f"Legacy spectral sheets are also incomplete: {missing_legacy_sheets}")

        H, wavelengths = _build_observation_matrix_from_legacy_sheets(
            xls,
            spectral_min,
            spectral_max,
            spectral_step,
            normalize,
            legacy_channel_order,
            use_abs)
        xls.close()
        return H, wavelengths

    led_df = pd.read_excel(xls, sheet_name='led_spectra')
    pd_df = pd.read_excel(xls, sheet_name='pd_responsivity')
    channel_df = pd.read_excel(xls, sheet_name='channel_map')
    prior_df = pd.read_excel(xls, sheet_name='optional_prior') if 'optional_prior' in xls.sheet_names else None
    xls.close()

    _require_columns(led_df, ['led_name', 'wavelength_nm', 'intensity'], 'led_spectra')
    _require_columns(pd_df, ['pd_name', 'wavelength_nm', 'responsivity'], 'pd_responsivity')
    _require_columns(channel_df, ['channel', 'led_name', 'pd_name', 'wavelength_min', 'wavelength_max'], 'channel_map')
    if prior_df is not None:
        _require_columns(prior_df, ['prior_name', 'wavelength_nm', 'value', 'value_type'], 'optional_prior')

    if spectral_min is None:
        spectral_min = float(channel_df['wavelength_min'].min())
    if spectral_max is None:
        spectral_max = float(channel_df['wavelength_max'].max())
    if spectral_step <= 0:
        raise ValueError('spectral_step must be positive')

    wavelengths = np.arange(float(spectral_min), float(spectral_max) + spectral_step * 0.5, spectral_step)
    channel_df = channel_df.sort_values('channel')
    H_rows = []

    for _, channel in channel_df.iterrows():
        led_name = channel['led_name']
        pd_name = channel['pd_name']
        wavelength_min = float(channel['wavelength_min'])
        wavelength_max = float(channel['wavelength_max'])

        led_curve = _interp_curve(led_df, 'led_name', led_name, 'intensity', wavelengths, 'led_spectra')
        pd_curve = _interp_curve(pd_df, 'pd_name', pd_name, 'responsivity', wavelengths, 'pd_responsivity')
        prior_curve = np.ones_like(wavelengths, dtype=float)

        if prior_df is not None and 'prior_name' in channel_df.columns and pd.notna(channel.get('prior_name')):
            prior_name = channel['prior_name']
            prior_curve = _interp_curve(
                prior_df,
                'prior_name',
                prior_name,
                'value',
                wavelengths,
                'optional_prior',
                outside_value=1.0)
            prior_type = prior_df[prior_df['prior_name'].astype(str) == str(prior_name)]['value_type'].dropna()
            if not prior_type.empty and str(prior_type.iloc[0]).lower() in ['absorbance', 'abs']:
                prior_curve = 10 ** (-prior_curve)

        band_mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
        row = led_curve * pd_curve * prior_curve
        row[~band_mask] = 0.0
        row = np.clip(row, 0.0, None)

        H_rows.append(_normalize_observation_row(row, wavelengths, normalize))

    return np.stack(H_rows, axis=0), wavelengths


def project_spectrum_to_observation(spectrum, H):
    """
    Project a generated high-dimensional spectrum back to LED-PD observations.

    spectrum shape:
        (n_samples, n_wavelengths)
        (n_samples, n_wavelengths, n_points)

    H shape:
        (n_channels, n_wavelengths)
    """
    spectrum = np.asarray(spectrum)
    H = np.asarray(H)

    if H.ndim != 2:
        raise ValueError('H must have shape (K, L)')
    if spectrum.ndim == 2:
        if spectrum.shape[1] != H.shape[1]:
            raise ValueError('spectrum wavelength dimension must match H')
        return np.einsum('kl,nl->nk', H, spectrum)
    if spectrum.ndim == 3:
        if spectrum.shape[1] != H.shape[1]:
            raise ValueError('spectrum wavelength dimension must match H')
        return np.einsum('kl,nlt->nkt', H, spectrum)
    raise ValueError('spectrum must have shape (N, L) or (N, L, T)')


def expand_observation_with_H(X_PPG, H):
    """
    Build a channel-weighted lifted representation from observed PPG and H.

    This is a deterministic representation expansion, not spectral recovery.
    X_PPG shape: (n_samples, n_channels, n_points)
    H shape: (n_channels, n_wavelengths)
    return shape: (n_samples, n_channels, n_points, n_wavelengths)
    """
    X_PPG = np.asarray(X_PPG)
    H = np.asarray(H)
    if X_PPG.ndim != 3:
        raise ValueError('X_PPG must have shape (N, K, T)')
    if H.ndim != 2:
        raise ValueError('H must have shape (K, L)')
    if X_PPG.shape[1] != H.shape[0]:
        raise ValueError('X_PPG channel count must match H channel count')
    return X_PPG[:, :, :, None] * H[None, :, None, :]

class Data_process:
    def __init__(self,file_path = "",num_files = 0,plot = 0):
        self.PPG_feature_extration = PPG_feature_extration()
        self.file_path = file_path
        self.num_files = num_files
        self.bg_index = 35
        self.data_raw = 0
        self.dataset_LED = 0
        self.dataset_Tem = 0
        self.dataset_Hum = 0
        self.dataset_p_info = 0
        self.dataset_BG = 0
        self.dataset_LED_feature = 0
        self.dataset_T_feature = 0
        self.dataset_H_feature = 0
        self.load_data_flag = 0
        self.initialize_data_flag = 0
        self.feature_extration_flag = 0
        self.scale_flag = 0
        self.ppg_upper_bond = 6000
        self.valley_location = []
        self.ac_dc_selected = []
        self.ac_dc_all = []
        self.top_k = 4
        self.plot = plot
        self.insufficient_rows = []  # 鐢ㄤ簬璁板綍鍒楁暟灏戜簬 self.top_k 鐨勮鐨勭储寮?

        self.BG_Min_Max =            [[3],   [30]]       # (min(dataset_BG), max(dataset_BG))
        self.LED_Min_Max =        [[[0], [600]],[[400], [6000]]]      # (min(dataset_LED), max(dataset_LED))
        self.T_Min_Max =             [[10],  [35]]       # (min(dataset_T), max(dataset_T))
        self.H_Min_Max =             [[30],  [100]]      # (min(dataset_H), max(dataset_H))
        self.p_info_Min_Max =        [[[0],  [1]],       # (min(gender), max(gender))
                                    [[18],  [70]],      # (min(age), max(age))
                                    [[150], [190]],     # (min(height), max(height))
                                    [[50],  [100]],     # (min(weight), max(weight))
                                    [[16],  [28]],      # (min(BMI), max(BMI))
                                    [[1],   [4]],      #  (min(finger_id), max(finger_id))
                                    [[0],   [2]],       # (min(DB_state), max(DB_state))
                                    [[0],   [200]]]     # (min(gender), max(gender))
        
        self.LED_feature_Min_Max =   [[[30], [120]],     # (min(hr), max(hr))
                                    [[80],  [100]]]     # (min(spo2), max(spo2)) ]
        self.LED_feature_Min_Max += self.LED_Min_Max * self.top_k * 6
        
        self.T_feature_Min_Max =     [[[15], [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[15],  [35]],      # (min(T), max(T))
                                    [[0],   [0.1]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [0.1]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [0.1]],     # (min(delta_T_3/3), max(delta_T_3/3))
                                    [[0],   [0.1]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [0.1]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [0.1]],     # (min(delta_T_3/3), max(delta_T_3/3))
                                    [[0],   [0.1]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [0.1]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [0.1]]]     # (min(delta_T_3/3), max(delta_T_3/3))
        
        self.H_feature_Min_Max =     [[[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[30], [100]],      # (min(H), max(H))
                                    [[0],   [4]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [4]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [4]],     # (min(delta_T_3/3), max(delta_T_3/3))
                                    [[0],   [4]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [4]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [4]],     # (min(delta_T_3/3), max(delta_T_3/3))
                                    [[0],   [4]],     # (min(delta_T_1/3), max(delta_T_1/3))
                                    [[0],   [4]],     # (min(delta_T_2/3), max(delta_T_2/3))
                                    [[0],   [4]]]     # (min(delta_T_3/3), max(delta_T_3/3))

    def load_data(self):
        print('Loading data ... ')
        file_name = []
        for _, _, filenames in os.walk(self.file_path):
            file_name.extend(filenames)
            break  # 鍙亶鍘嗛《灞傜洰褰?
        
        data_frames = []
        for index in range(0, min(self.num_files, len(file_name))):
            file_path = os.path.join(self.file_path, file_name[index])
            encodings = ['utf-8', 'GBK', 'ISO-8859-1', 'cp1252']  # 鍒楄〃鍙兘闇€瑕佹牴鎹綘鐨勯渶姹傝皟鏁?
            loaded = False
            for encoding in encodings:
                try:
                    data_temp = pd.read_csv(file_path, header=0, encoding=encoding)
                    data_frames.append(data_temp)
                    print(f"Loaded {file_name[index]} with encoding {encoding}")
                    loaded = True
                    break
                except UnicodeDecodeError as e:
                    pass
                    # print(f"Error loading {file_name[index]} with {encoding}: {e}")
            
            if not loaded:
                print(f"Failed to load {file_name[index]} with any known encoding")
        
        if data_frames:
            self.data_raw = pd.concat(data_frames, ignore_index=True)
            print('Done')
            self.load_data_flag = 1
        else:
            print("No data loaded.")
        
    def data_modify(self, enable_default = False):
        print("Debug info: Type of self.data_raw before any operation:", type(self.data_raw))

        data = self.data_raw
        meal_interval_col = '据上一次用餐已过时间(min)'
        data[meal_interval_col] = pd.to_numeric(data[meal_interval_col], errors='coerce')

        conditions = [
            ((data[meal_interval_col] < 15) | (data[meal_interval_col] >= 300)),
            ((data[meal_interval_col] >= 15) & (data[meal_interval_col] < 45)),
            ((data[meal_interval_col] >= 45) & (data[meal_interval_col] < 75)),
            ((data[meal_interval_col] >= 75) & (data[meal_interval_col] < 105)),
            ((data[meal_interval_col] >= 105) & (data[meal_interval_col] < 135)),
            ((data[meal_interval_col] >= 135) & (data[meal_interval_col] < 300))
        ]
        choices = ['空腹', '餐后30分钟', '餐后60分钟', '餐后90分钟', '餐后120分钟','其他']

        # 使用np.select为数据行分配时间段标签
        data['时间段'] = np.select(conditions, choices, default='其他')
                                
        # 确保日期列是日期时间格式
        data['日期'] = pd.to_datetime(data['日期'])

        # 计算每个日期所在的季度
        data['季度'] = data['日期'].dt.to_period('Q')

        # 按ID, 分组标识符和时间段分组，计算最小值和最大值
        group_stats = data.groupby(['ID', '季度', '时间段'])['平均值'].agg(['min', 'max']).reset_index()

        # 重命名统计数据以便合并
        group_stats.rename(columns={'min': '分组最小值', 'max': '分组最大值'}, inplace=True)

        new_columns_added = ['时间段','季度']
        for choice in choices:
            prefix = choice.replace("", "")
            temp_stats = group_stats[group_stats['时间段'] == choice]
            temp_stats = temp_stats.rename(columns={
                '分组最小值': f'{prefix}最小值',
                '分组最大值': f'{prefix}最大值'
            })
            new_columns_added.extend([f'{prefix}最小值', f'{prefix}最大值'])
            data = pd.merge(
                data,
                temp_stats[['ID', '季度', f'{prefix}最小值', f'{prefix}最大值']],
                on=['ID', '季度'],
                how='left')

        insert_position_start = data.columns.get_loc('信号质量') + 1
        columns_except_new = [col for col in data.columns if col not in new_columns_added]
        columns_before_insertion = columns_except_new[:insert_position_start]
        columns_after_insertion = columns_except_new[insert_position_start:]
        data = data[columns_before_insertion + new_columns_added + columns_after_insertion]

        preset_values = {
            ('正常', '空腹最小值'): 4,
            ('正常', '空腹最大值'): 6,
            ('正常', '餐后30分钟最小值'): 6,
            ('正常', '餐后30分钟最大值'): 8,
            ('正常', '餐后60分钟最小值'): 6,
            ('正常', '餐后60分钟最大值'): 11,
            ('正常', '餐后90分钟最小值'): 6,
            ('正常', '餐后90分钟最大值'): 9,
            ('正常', '餐后120分钟最小值'): 5,
            ('正常', '餐后120分钟最大值'): 8,
            ('正常', '其他最小值'): 4,
            ('正常', '其他最大值'): 6,
            
            ('1型糖尿病', '空腹最小值'): 5,
            ('1型糖尿病', '空腹最大值'): 8,
            ('1型糖尿病', '餐后30分钟最小值'): 6,
            ('1型糖尿病', '餐后30分钟最大值'): 13,
            ('1型糖尿病', '餐后60分钟最小值'): 7,
            ('1型糖尿病', '餐后60分钟最大值'): 15,
            ('1型糖尿病', '餐后90分钟最小值'): 8,
            ('1型糖尿病', '餐后90分钟最大值'): 15,
            ('1型糖尿病', '餐后120分钟最小值'): 8,
            ('1型糖尿病', '餐后120分钟最大值'): 15,
            ('1型糖尿病', '其他最小值'): 6,
            ('1型糖尿病', '其他最大值'): 10,
            
            ('2型糖尿病', '空腹最小值'): 5,
            ('2型糖尿病', '空腹最大值'): 8,
            ('2型糖尿病', '餐后30分钟最小值'): 6,
            ('2型糖尿病', '餐后30分钟最大值'): 13,
            ('2型糖尿病', '餐后60分钟最小值'): 7,
            ('2型糖尿病', '餐后60分钟最大值'): 15,
            ('2型糖尿病', '餐后90分钟最小值'): 8,
            ('2型糖尿病', '餐后90分钟最大值'): 15,
            ('2型糖尿病', '餐后120分钟最小值'): 8,
            ('2型糖尿病', '餐后120分钟最大值'): 15,
            ('2型糖尿病', '其他最小值'): 6,
            ('2型糖尿病', '其他最大值'): 10,
            # 为其他确诊情况和列组合设置预设值...
        }

        columns_to_check = [f'{prefix}最小值' for prefix in choices] + [f'{prefix}最大值' for prefix in choices]
        nan_exists = data[columns_to_check].isna().any().any()

        if enable_default:
            for (condition, column), preset_value in preset_values.items():
                data[column] = preset_value
        else:
            if nan_exists:
                print("存在NaN值")
                for (condition, column), preset_value in preset_values.items():
                    condition_rows = data['确诊情况'] == condition
                    for column in columns_to_check:
                        data.loc[condition_rows, column] = data.loc[condition_rows, column].fillna(preset_value)
            else:
                print("不存在NaN值")

        if nan_exists:
            print("存在NaN值")
            nan_rows = data[columns_to_check].isna().any(axis=1)
            nan_indices = data[nan_rows].index
            print(nan_indices)
        else:
            print("不存在NaN值")

        self.data_raw = data

    def label_trans(self,input):
        for i in range(np.shape(input)[0]):
            if input[i,4] == '男': input[i,4] = 0
            else: input[i,4] = 1
            
            if input[i,9] == '1型糖尿病': input[i,9] = 1
            elif input[i,9] == '2型糖尿病': input[i,9] = 1
            else: input[i,9] = 0
            
            if input[i,12] > 200 or input[i,12] < 0: input[i,12] = 0       
            
            input[i,14] = 3
    
    def data_export(self, modify=True):
        self.load_data()
        self.data_modify()
        self.initialize_data()
        self.feature_extration()
        data_p = self.data_raw[:,:self.bg_index]
        self.label_trans(data_p)
            
        X_test_modal1 = self.dataset_LED_feature
        X_test_modal2 = self.dataset_T_feature
        X_test_modal3 = self.dataset_H_feature
        X_test_modal4 = self.dataset_p_info[:,:-1]
        X_test_modal5 = self.dataset_p_info[:,-1]
        X_test_modal6 = self.data_raw[:,[18,19,22,23]]
        
        print(np.shape(X_test_modal1))
        print(np.shape(X_test_modal2))
        print(np.shape(X_test_modal3))
        print(np.shape(X_test_modal4))
        print(np.shape(X_test_modal5))
        print(np.shape(X_test_modal6))
        print(np.shape(data_p))
        
        data_export = np.hstack((data_p,X_test_modal1,X_test_modal2,X_test_modal3,X_test_modal4,X_test_modal5,X_test_modal6))
        return data_export
    
    def initialize_data(self, enable_reverse = False):
        if self.load_data_flag == 1:
            print('Initializing data ... ')
            self.data_raw = np.array(self.data_raw)
            if enable_reverse:
                self.dataset_LED =   self.ppg_upper_bond - self.data_raw[:,self.bg_index + 1 +1800         :self.bg_index + 1 +1800*2].reshape(-1,6,300)     # 6个波段的PPG数据，每段300采样点
            else:
                self.dataset_LED =   self.data_raw[:,self.bg_index + 1 +1800         :self.bg_index + 1 +1800*2].reshape(-1,6,300)     # 6个波段的PPG数据，每段300采样点
                
            self.dataset_Tem =       self.data_raw[:,self.bg_index + 1 +1800*2       :self.bg_index + 1 +1800*2+156].reshape(-1,3,52)                # 温度数据， 每段52采样点
            self.dataset_Hum =       self.data_raw[:,self.bg_index + 1 +1800*2+156   :self.bg_index + 1 +1800*2+156*2].reshape(-1,3,52)              # 湿度数据， 每段52采样点
            self.dataset_p_info =    self.data_raw[:,[4,5,6,7,8,14,9,12]]                           # 其他数据， 性别4 年龄5 身高6 体重7 BMI8 测量部位14 确诊状况9 用餐间隔12
            self.dataset_bg_info =   self.data_raw[:,18:30]                           
            self.dataset_BG  =       self.data_raw[:,self.bg_index]                                            # 血糖数据， mmol/L
            self.label_transfer(self.dataset_p_info)  # 其他数据数值化
            print('Done')
            self.initialize_data_flag = 1
        else:
            print('Please load data using your_object_name.load_data()!')        
    
    def resample_data(self, data, new_length):
        n_samples, old_length, n_channels = data.shape
        new_data = np.zeros((n_samples, new_length, n_channels))
        
        for i in range(n_samples):
            for j in range(n_channels):
                x_old = np.linspace(0, 1, old_length)
                x_new = np.linspace(0, 1, new_length)
                f = interp1d(x_old, data[i, :, j], kind='linear')
                new_data[i, :, j] = f(x_new)
        
        return new_data

    def compute_vpg_apg(self, ppg_signals):
        """
        计算每个通道的VPG（一次微分）和APG（二次微分）。

        参数:
        ppg_signals (numpy.ndarray): 形状为 (n_samples, n_points, n_channels) 的PPG波形数据。

        返回:
        ppg_vpg_apg (list): 每个元素是形状为 (n_samples, n_points, 3) 的数组，包含原始PPG、VPG和APG。
        """
        # 初始化存储VPG和APG的数组
        vpg = np.zeros_like(ppg_signals)
        apg = np.zeros_like(ppg_signals)

        # 计算每个通道的VPG和APG
        for i in range(ppg_signals.shape[2]):
            for j in range(ppg_signals.shape[0]):
                ppg_signal = ppg_signals[j, :, i]
                vpg_signal = np.gradient(ppg_signal)
                apg_signal = np.gradient(vpg_signal)
                
                vpg[j, :, i] = vpg_signal
                apg[j, :, i] = apg_signal

        return vpg, apg
  
    def wavelet_denoising(self, data, wavelet='db4', level=1):
        # 分解
        coeffs = pywt.wavedec(data, wavelet, mode="per", level=level)
        # 计算阈值
        threshold = np.sqrt(2*np.log(len(data))) * np.median(np.abs(coeffs[-level])) / 0.6745
        # 阈值处理
        coeffs[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:]]
        # 重构
        return pywt.waverec(coeffs, wavelet, mode='per')

    def baseline_correction(self, data, window_length=49, polyorder=3):
        if window_length % 2 == 0 or window_length > len(data):
            window_length = min(len(data) - 1, window_length)
            if window_length % 2 == 0:
                window_length -= 1
        baseline = savgol_filter(data, window_length, polyorder)
        return data - baseline + np.mean(baseline)
    
    def PPG_smooth(self, data):
        denoised_data = np.zeros_like(data)
        # 遍历每个样本的每个通道
        for i in range(data.shape[0]):  # 遍历样本
            for j in range(data.shape[1]):  # 遍历通道
                # 对每个信号应用小波去噪
                temp = self.wavelet_denoising(data[i, j, :], wavelet='db4', level=1)
                denoised_data[i, j, :] = self.baseline_correction(temp, window_length=49, polyorder=3)
        return denoised_data

    def feature_extration(self, enable_smooth = False,enable_top_k = False):
        if self.initialize_data_flag == 1:
            print('Feature extrating ... ')
            self.insufficient_rows = []
            if enable_smooth:
                self.dataset_LED = self.PPG_smooth(self.dataset_LED)
            self.dataset_LED_feature = self.pre_calculation_PPG(self.dataset_LED, enable_top_k=enable_top_k)  # 心率、血氧、PPG峰谷数据
            self.dataset_T_feature = self.pre_calculation_TH(self.dataset_Tem)  # 心率、血氧、PPG峰谷数据
            self.dataset_H_feature = self.pre_calculation_TH(self.dataset_Hum)  # 心率、血氧、PPG峰谷数据
            if self.insufficient_rows:
                self.insufficient_rows_remove()
            self.invalid_rows_remove()
            self.feature_extration_flag = 1
            print('Done')
        elif self.load_data_flag == 0: print('Please load data using your_object_name.load_data()!!') 
        elif self.initialize_data_flag == 0: print('Please initialize data using your_object_name.initialize_data()!!')

    def insufficient_rows_remove(self):
        if not self.insufficient_rows:
            print('No insufficient rows to remove.')
        else:
            print(f'Removing rows: {self.insufficient_rows}')
            # 使用 np.delete 从数组中删除指定的行
            self.data_raw = np.delete(self.data_raw, self.insufficient_rows, axis=0)
            self.dataset_LED = np.delete(self.dataset_LED, self.insufficient_rows, axis=0)
            self.dataset_Tem = np.delete(self.dataset_Tem, self.insufficient_rows, axis=0)
            self.dataset_Hum = np.delete(self.dataset_Hum, self.insufficient_rows, axis=0)
            self.dataset_bg_info = np.delete(self.dataset_bg_info, self.insufficient_rows, axis=0)
            self.dataset_BG = np.delete(self.dataset_BG, self.insufficient_rows, axis=0)
            self.dataset_p_info = np.delete(self.dataset_p_info, self.insufficient_rows, axis=0)
            self.dataset_LED_feature = np.delete(self.dataset_LED_feature, self.insufficient_rows, axis=0)
            self.dataset_T_feature = np.delete(self.dataset_T_feature, self.insufficient_rows, axis=0)
            self.dataset_H_feature = np.delete(self.dataset_H_feature, self.insufficient_rows, axis=0)
            self.insufficient_rows = []

    def invalid_rows_remove(self):
        arrays_to_check = [
            self.dataset_LED,
            self.dataset_Tem,
            self.dataset_Hum,
            self.dataset_bg_info,
            self.dataset_BG,
            self.dataset_p_info,
            self.dataset_LED_feature,
            self.dataset_T_feature,
            self.dataset_H_feature,
        ]
        rows_to_remove = set()

        for data in arrays_to_check:
            data_temp = np.asarray(data, dtype=float)
            if data_temp.ndim == 1:
                invalid_mask = ~np.isfinite(data_temp)
            else:
                invalid_mask = ~np.isfinite(data_temp).all(axis=tuple(range(1, data_temp.ndim)))
            rows_to_remove.update(np.where(invalid_mask)[0].tolist())

        led_feature = np.asarray(self.dataset_LED_feature[:, :2], dtype=float)
        invalid_hr_spo2 = np.any(led_feature <= 0, axis=1)
        rows_to_remove.update(np.where(invalid_hr_spo2)[0].tolist())

        rows_to_remove = sorted(rows_to_remove)
        if rows_to_remove:
            print(f'Removing invalid rows: {rows_to_remove}')
            self.data_raw = np.delete(self.data_raw, rows_to_remove, axis=0)
            self.dataset_LED = np.delete(self.dataset_LED, rows_to_remove, axis=0)
            self.dataset_Tem = np.delete(self.dataset_Tem, rows_to_remove, axis=0)
            self.dataset_Hum = np.delete(self.dataset_Hum, rows_to_remove, axis=0)
            self.dataset_bg_info = np.delete(self.dataset_bg_info, rows_to_remove, axis=0)
            self.dataset_BG = np.delete(self.dataset_BG, rows_to_remove, axis=0)
            self.dataset_p_info = np.delete(self.dataset_p_info, rows_to_remove, axis=0)
            self.dataset_LED_feature = np.delete(self.dataset_LED_feature, rows_to_remove, axis=0)
            self.dataset_T_feature = np.delete(self.dataset_T_feature, rows_to_remove, axis=0)
            self.dataset_H_feature = np.delete(self.dataset_H_feature, rows_to_remove, axis=0)
            
    def data_scale(self):
        if self.scale_flag == 0:
            print('Data scaling ... ')
            self.dataset_LED = self.scaler(self.dataset_LED, self.LED_Min_Max[1]).reshape(-1,6,300)             
            self.dataset_Tem = self.scaler(self.dataset_Tem, self.T_Min_Max).reshape(-1,3,52)
            self.dataset_Hum = self.scaler(self.dataset_Hum, self.H_Min_Max).reshape(-1,3,52)
            self.dataset_bg_info = self.scaler(self.dataset_bg_info, self.BG_Min_Max).reshape(-1,12)
            self.dataset_BG = self.scaler(self.dataset_BG, self.BG_Min_Max).reshape(-1,1)
            for i in range(np.shape(self.dataset_p_info)[1]):
                self.dataset_p_info[:,i] = self.scaler(self.dataset_p_info[:,i], self.p_info_Min_Max[i]).reshape(1,-1)
            for i in range(np.shape(self.dataset_LED_feature)[1]):
                self.dataset_LED_feature[:,i] = self.scaler(self.dataset_LED_feature[:,i], self.LED_feature_Min_Max[i]).reshape(1,-1)
            for i in range(np.shape(self.dataset_T_feature)[1]):
                self.dataset_T_feature[:,i] = self.scaler(self.dataset_T_feature[:,i], self.T_feature_Min_Max[i]).reshape(1,-1)
            for i in range(np.shape(self.dataset_H_feature)[1]):
                self.dataset_H_feature[:,i] = self.scaler(self.dataset_H_feature[:,i], self.H_feature_Min_Max[i]).reshape(1,-1) 
            self.scale_flag = 1
            print('Done')
        elif self.load_data_flag == 0: print('Please do data loading using your_object_name.load_data()!!') 
        elif self.initialize_data_flag == 0: print('Please do data initialization using your_object_name.initialize_data()!!')
        elif self.feature_extration_flag == 0: print('Please do feature extration using your_object_name.feature_extration()!!')
        elif self.scale_flag == 0: print('Data already scaled')

    def pre_calculation_PPG(self, data_in, enable_top_k=False):  # PPG预处理
        data_out = []
        feature_len = 2 + (self.PPG_feature_extration.n_led * self.top_k * 2 if enable_top_k else self.PPG_feature_extration.n_led * 2)
        invalid_feature = [-999, -999] + [0] * (feature_len - 2)

        def mark_insufficient_row(row_index, reason):
            if row_index not in self.insufficient_rows:
                self.insufficient_rows.append(row_index)
            print(f"Row {row_index} skipped: {reason}")
        
        for i in range(np.shape(data_in)[0]):
            ppg_result = self.PPG_feature_extration.maxim_heart_rate_and_oxygen_saturation(data_in[i,5,:], data_in[i,4,:])
            if not isinstance(ppg_result, tuple) or len(ppg_result) != 4:
                mark_insufficient_row(i, "invalid PPG peak result")
                data_out.append(invalid_feature.copy())
                continue

            hr_val, spo2_val, self.valley_location, r_value = ppg_result
            if len(self.valley_location) < 2:
                mark_insufficient_row(i, f"only {len(self.valley_location)} valid PPG valleys")
                data_out.append(invalid_feature.copy())
                continue

            self.ac_dc_selected, self.ac_dc_all = self.PPG_feature_extration.AC_DC_value(data_in[i,0,:], data_in[i,1,:], data_in[i,2,:], data_in[i,3,:], data_in[i,4,:], data_in[i,5,:], self.valley_location, self.plot)
            
            if enable_top_k:
                if np.shape(self.ac_dc_all)[1] >= self.top_k:
                    # 使用argsort找出指定维度上的最大的top_k个元素的索引
                    indices = np.argsort(self.ac_dc_all[:,:,1], axis=1)[:,-self.top_k:][:, ::-1]  # 对每一行取最大的top_k个值，并逆序排列
                    ac_dc_all_top_k = np.take_along_axis(self.ac_dc_all, indices[:,:,np.newaxis], axis=1)
                    data_out.append([hr_val] + [spo2_val] + list(ac_dc_all_top_k.flatten()))
                else:
                    mark_insufficient_row(i, f"only {np.shape(self.ac_dc_all)[1]} AC/DC segments, need {self.top_k}")
                    data_out.append(invalid_feature.copy())
            else:
                data_out.append([hr_val] + [spo2_val] + list(self.ac_dc_selected))
        
        return np.array(data_out)

    def pre_calculation_TH(self,data_in): # 温湿度预处理
        data_out = []
        for i in range(np.shape(data_in)[0]):
            data_temp = self.PPG_feature_extration.TH_preprocess(data_in[i])
            data_out.append(data_temp)
        return np.array(data_out)

    def scaler(self,data_in, bound): # 归一化
        scaler_temp = MinMaxScaler().fit(bound)
        data_out = scaler_temp.transform(data_in.reshape(-1, 1))
        return data_out

    def label_transfer(self,input): # 文本数据数字化
        for i in range(np.shape(input)[0]):
            if input[i,0] == '男': input[i,0] = 0
            else: input[i,0] = 1

            input[i,5] = 3

            if input[i,6] == '1型糖尿病': input[i,6] = 1
            elif input[i,6] == '2型糖尿病': input[i,6] = 1
            else: input[i,6] = 0

            if input[i,7] > 200 or input[i,7] < 0: input[i,7] = 0

class PPG_feature_extration:
    def __init__(self):
        # self.pn_npks = 0
        self.max_n_peaks = 50
        self.min_height = 0
        
        self.len_LED, self.len_TH = 150, 13
        self.pn_spo2, self.pch_spo2_valid = 0, 0
        self.pn_heart_rate, self.pch_hr_valid = 0, 0
        self.auw_hamm = [40.96,276.48,512,276.48,40.96]
        self.FS = 50
        self.n_led = 6
        self.BUFFER_SIZE = int(self.len_LED*2)
        self.MA4_SIZE = 4
        self.HAMMING_SIZE = 5
        self.peak_distance = 20
        
        self.r, self.sigma = 11, 2

    def maxim_find_peaks(self, pn_x, n_size):
        self.maxim_peaks_above_min_height(pn_x, n_size)
        self.maxim_remove_close_peaks(pn_x)
        self.pn_npks = min(self.pn_npks, self.max_n_peaks)

    def maxim_peaks_above_min_height(self, pn_x, n_size):
        i = 1
        n_width = 0
        while (i < n_size -1):
            if (pn_x[i] > self.min_height and pn_x[i] > pn_x[i-1]): #find left edge of potential peaks
                n_width = 1
                while (i + n_width < n_size and pn_x[i] == pn_x[i+n_width]): #find flat peaks
                    n_width += 1
                if (pn_x[i] > pn_x[i+n_width] and self.pn_npks < self.max_n_peaks ): #find right edge of peaks
                    self.pn_locs[self.pn_npks] = i
                    self.pn_npks = self.pn_npks + 1
                    i += n_width + 1
                else:
                    i += n_width
            else:
                i += 1

    def maxim_remove_close_peaks(self, pn_x):
        i, j, n_old_npks, n_dist = 0, 0, 0, 0
        self.maxim_sort_indices_descend(pn_x, self.pn_locs, self.pn_npks) # Order peaks from large to small
        # for i in range(-1,pn_npks):
        i = -1
        while i < self.pn_npks:
            n_old_npks = self.pn_npks
            self.pn_npks = i + 1
            for j in range (i+1,n_old_npks):
                if i == -1:
                    n_dist =  self.pn_locs[j] - (-1)
                else:
                    n_dist =  self.pn_locs[j] - self.pn_locs[i]
                # lag-zero peak of autocorr is at index -1
                if (abs(n_dist) > self.peak_distance):
                    self.pn_locs[self.pn_npks] = self.pn_locs[j]
                    self.pn_npks = self.pn_npks + 1 
            i += 1
        self.maxim_sort_ascend(self.pn_locs, self.pn_npks)####修改
        self.pn_locs = self.pn_locs[:self.pn_npks]

    def maxim_sort_ascend(self, pn_x, n_size): 
        for i in range(1,n_size):
            n_temp = pn_x[i]
            j = i - 1
            while j >=0 and n_temp < pn_x[j]:
                pn_x[j+1] = pn_x[j]
                j = j - 1
            pn_x[j+1] = n_temp

    def maxim_sort_indices_descend(self, pn_x, pn_indx, n_size):
        for i in range(1,n_size):
            n_temp = pn_indx[i]
            j = i - 1
            while j >=0 and (pn_x[int(n_temp)] > pn_x[int(pn_indx[j])]):
                pn_indx[j+1] = pn_indx[j]
                j = j - 1
            pn_indx[j+1] = n_temp

    def maxim_heart_rate_and_oxygen_saturation(self, S5, S6):
        self.pn_npks = 0
        self.pn_locs = np.zeros(self.max_n_peaks)
        self.min_height = 0
        an_x = np.zeros(self.BUFFER_SIZE)
        an_y = np.zeros(self.BUFFER_SIZE)

        # remove DC of ir signal
        un_ir_mean = 0
        for k in range (0,self.BUFFER_SIZE):
            un_ir_mean += S5[k]
        un_ir_mean = un_ir_mean/self.BUFFER_SIZE
        for k in range (0,self.BUFFER_SIZE):
            an_x[k] =  S5[k] - un_ir_mean

        # 4 pt Moving Average
        # n_denom = 0
        # for k in range (0,self.BUFFER_SIZE-self.MA4_SIZE):
        #     n_denom = an_x[k] + an_x[k+1] + an_x[k+2]+ an_x[k+3]
        #     an_x[k] = n_denom/4

        # 一阶差分
        an_dx = np.zeros(self.BUFFER_SIZE)
        for k in range(0,self.BUFFER_SIZE-1):
            an_dx[k]= an_x[k+1] - an_x[k]

        # 2-pt moving average
        for k in range (0,self.BUFFER_SIZE-2):
            n_denom = an_dx[k] + an_dx[k+1]
            an_dx[k] = n_denom/2
        
        # hamming window构造一个函数。这个函数在某一区间有非零值，而在其余区间皆为0.汉明窗就是这样的一种函数
        # flip wave form so that we can detect valley with peak detector
        # 翻转波形，就能利用波峰探测器探测到波谷
        for i in range(0,self.BUFFER_SIZE-self.HAMMING_SIZE-2):
            # print(i)
            s = 0
            for k in range (i,i + self.HAMMING_SIZE):
                s -= an_dx[k] *self.auw_hamm[k-i]
            an_dx[i]= s/sum(self.auw_hamm) #divide by sum of auw_hamm 
        
        # n_th1 = 0 #threshold calculation阈值计算
        for k in range(0,self.BUFFER_SIZE-self.HAMMING_SIZE):
            if an_dx[k] > 0:
                self.min_height += an_dx[k]
            else:
                self.min_height += -an_dx[k]
        self.min_height= self.min_height/ ( self.BUFFER_SIZE-self.HAMMING_SIZE)

        # peak location is acutally index for sharpest location of raw signal since we flipped the signal
        # an_dx_peak_locs = np.zeros(max_num_peaks)

        self.maxim_find_peaks(an_dx, self.BUFFER_SIZE-self.HAMMING_SIZE) #peak_height, peak_distance, max_num_peaks 
        # print('an_dx_peak_locs',an_dx_peak_locs)
        n_peak_interval_sum = 0
        if self.pn_npks >= 2: 
            for k in range (1,self.pn_npks):
                n_peak_interval_sum += self.pn_locs[k] - self.pn_locs[k-1]
            n_peak_interval_sum = n_peak_interval_sum/(self.pn_npks - 1)
            self.pn_heart_rate = self.FS * 60 / n_peak_interval_sum #计算出脉率-- beats per minutes
            self.pn_heart_rate = round(self.pn_heart_rate,2)
            self.pch_hr_valid = 1
        else: #波谷小于2个无法计算hr
            self.pn_heart_rate = -999
            self.pch_hr_valid = 0
        
        #初始数据波谷位置修正
        an_ir_valley_locs = np.zeros(self.pn_npks)
        for k in range (0,self.pn_npks):
            an_ir_valley_locs[k] = self.pn_locs[k] + int(self.HAMMING_SIZE/2) + 1 + 1
        # print('an_ir_valley_locs',an_ir_valley_locs)

        # raw value : RED(=y) and IR(=x)
        # we need to assess DC and AC value of ir and red PPG. 
        for k in range (0,self.BUFFER_SIZE):
            an_x[k] = S5[k]
            an_y[k] = S6[k]

        #find precise min near an_ir_valley_locs 精确的查找位置减小spo2误差
        n_exact_ir_valley_locs_count = 0
        m = 0
        n_c_min = 0
        an_exact_ir_valley_locs = np.zeros(self.pn_npks)
        for k in range(0,self.pn_npks):
            un_only_once = 1
            m = int(an_ir_valley_locs[k])
            n_c_min = 2**24
            if (m + 5 <  self.BUFFER_SIZE-self.HAMMING_SIZE and m - 5 > 0):
                for i in range (m-5,m+5):
                    if an_x[i] < n_c_min:
                        if un_only_once > 0:
                            un_only_once = 0
                        n_c_min = an_x[i]
                        an_exact_ir_valley_locs[k] = i
                if un_only_once == 0:
                    n_exact_ir_valley_locs_count += 1
        an_exact_ir_valley_locs = an_exact_ir_valley_locs[:n_exact_ir_valley_locs_count]##修改

        if n_exact_ir_valley_locs_count < 2: #波谷小于2个无法计算hr
            self.pn_spo2 = -999
            self.pch_spo2_valid = 0
            return self.pn_heart_rate, self.pn_spo2, an_exact_ir_valley_locs, 0
        
        # 4-pt moving average
        # for k in range(0,self.BUFFER_SIZE-self.MA4_SIZE):
        #     an_x[k]=(an_x[k]+an_x[k+1]+ an_x[k+2]+ an_x[k+3])/4
        #     an_y[k]=(an_y[k]+an_y[k+1]+ an_y[k+2]+ an_y[k+3])/4

        # using an_exact_ir_valley_locs , find ir-red DC andir-red AC for SPO2 calibration ratio
        # finding AC/DC maximum of raw ir * red between two valley locations
        n_ratio_average = 0
        n_i_ratio_count = 0
        an_ratio = np.zeros(self.pn_npks)

        for k in range (0,n_exact_ir_valley_locs_count):
            if an_exact_ir_valley_locs[k] > self.BUFFER_SIZE:
                self.pn_spo2 = -999 #do not use SPO2 since valley loc is out of range
                self.pch_spo2_valid = 0
                return self.pn_heart_rate, self.pn_spo2, an_exact_ir_valley_locs, 0

        # find max between two valley locations 
        # and use ratio betwen AC compoent of Ir & Red and DC compoent of Ir & Red for SPO2 
        n_x_dc_max, n_x_dc_max_idx, n_x_ac = 0, 0, 0
        n_y_dc_max, n_y_dc_max_idx, n_y_ac = 0, 0, 0
        n_nume = 0
        # n_x_valley_idx = 0
        # valley_locs = np.zeros(self.pn_npks-1)
        # print('an_exact_ir_valley_locs',an_exact_ir_valley_locs)
        
        # for k in range (0,n_exact_ir_valley_locs_count-1):
        #     n_x_valley = 2**24
        #     if an_exact_ir_valley_locs[k+1] - an_exact_ir_valley_locs[k] > self.peak_distance:
        #         for i in range(int(an_exact_ir_valley_locs[k]), int(an_exact_ir_valley_locs[k+1])):
        #             if an_x[i] < n_x_valley:
        #                 n_x_valley = an_x[i]
        #                 # print(an_x[i],'<',n_x_valley)
        #                 n_x_valley_idx = i
        #         valley_locs[k] = n_x_valley_idx

        # an_exact_ir_valley_locs = valley_locs
        # n_exact_ir_valley_locs_count = n_exact_ir_valley_locs_count - 1
        # print('an_exact_ir_valley_locs',an_exact_ir_valley_locs)
        
        for k in range (0,n_exact_ir_valley_locs_count-1):
            n_y_dc_max = - 2**24
            n_x_dc_max = - 2**24

            if an_exact_ir_valley_locs[k+1] - an_exact_ir_valley_locs[k] > self.peak_distance:
                for i in range(int(an_exact_ir_valley_locs[k]), int(an_exact_ir_valley_locs[k+1])):
                    if an_x[i] > n_x_dc_max:    #IR max
                        n_x_dc_max = an_x[i]
                        n_x_dc_max_idx = i

                    if an_y[i] > n_y_dc_max:    #Red max
                        n_y_dc_max = an_y[i]
                        n_y_dc_max_idx = i 
                # print(n_x_dc_max)
                # print(n_y_dc_max)
                # 利用f(t)=dc(t)+ac(t) 分别求dc和ac
                n_y_ac = (an_y[int(an_exact_ir_valley_locs[k+1])] - an_y[int(an_exact_ir_valley_locs[k])])*(n_y_dc_max_idx -an_exact_ir_valley_locs[k])
                n_y_ac =  an_y[int(an_exact_ir_valley_locs[k])] + n_y_ac/ (an_exact_ir_valley_locs[k+1] - an_exact_ir_valley_locs[k])
                n_y_ac =  an_y[n_y_dc_max_idx] - n_y_ac # subtracting linear DC compoenents from raw

                # print((an_y[int(an_exact_ir_valley_locs[k+1])] - an_y[int(an_exact_ir_valley_locs[k])]))
                n_x_ac = (an_x[int(an_exact_ir_valley_locs[k+1])] - an_x[int(an_exact_ir_valley_locs[k])])*(n_x_dc_max_idx -an_exact_ir_valley_locs[k])
                n_x_ac =  an_x[int(an_exact_ir_valley_locs[k])] + n_x_ac/ (an_exact_ir_valley_locs[k+1] - an_exact_ir_valley_locs[k])
                n_x_ac =  an_x[n_x_dc_max_idx] - n_x_ac # subtracting linear DC compoenents from raw

                n_nume = n_y_ac * n_x_dc_max
                n_denom = n_x_ac *n_y_dc_max

                if (n_denom > 0 and n_i_ratio_count < self.pn_npks and n_nume != 0):
                    an_ratio[n_i_ratio_count] = n_nume/n_denom    #R = ( n_y_ac *n_x_dc_max) / ( n_x_ac *n_y_dc_max)
                    n_i_ratio_count += 1


        self.maxim_sort_ascend(an_ratio, n_i_ratio_count)
        n_middle_idx = 0
        n_middle_idx = int(n_i_ratio_count/2)

        # print(an_ratio)
        # print(n_middle_idx)

        if n_middle_idx > 1:
            n_ratio_average = (an_ratio[n_middle_idx-1] + an_ratio[n_middle_idx])/2 # use median R value
        else:
            n_ratio_average = an_ratio[n_middle_idx]

        if (n_ratio_average > 0 and n_ratio_average < 1.84): 
            self.pch_spo2_valid = 1
            self.pn_spo2 = -45.060*n_ratio_average**2 + 30.354 *n_ratio_average + 94.845
            self.pn_spo2 = round(self.pn_spo2,2)
            # print('n_ratio_average = ', n_ratio_average)
        else:
            self.pn_spo2 = -999

        return self.pn_heart_rate, self.pn_spo2, an_exact_ir_valley_locs, n_ratio_average
    
    def AC_DC_value(self, S1, S2, S3, S4, S5, S6, val_loc, plot_flag):
        
        an_S = np.zeros((6,self.BUFFER_SIZE))

        for k in range (0,self.BUFFER_SIZE):
            an_S[0,k] = S1[k]
            an_S[1,k] = S2[k]
            an_S[2,k] = S3[k]
            an_S[3,k] = S4[k]
            an_S[4,k] = S5[k]
            an_S[5,k] = S6[k]

        n_i_ratio_count = 0
        S_AC_DC = np.zeros((12,len(val_loc)))
        an_ratio = np.zeros(len(val_loc))
        ac_dc_location = np.zeros((6,len(val_loc)-1,2))

        # find max between two valley locations 
        # and use ratio betwen AC compoent of Ir & Red and DC compoent of Ir & Red for SPO2 
        n_S_dc_max = [0] * 6
        n_S_dc_max_idx = [0] * 6
        n_S_ac = [0] * 6

        if plot_flag:
            fig_test, ax = plt.subplots(figsize=(10,10),nrows = 6, ncols = 1)       # for plot
            for i in range(6):                                                      # for plot
                ax[i].plot(an_S[i])                                                 # for plot

        n_nume = 0
        for k in range (0,len(val_loc)-1):
            n_S_dc_max = [- 2**24] * 6

            if val_loc[k+1] - val_loc[k] > self.peak_distance:
                for i in range(int(val_loc[k]), int(val_loc[k+1])):
                    for j in range(6):
                        if an_S[j,i] > n_S_dc_max[j]:    #S1 max
                            n_S_dc_max[j] = an_S[j,i]
                            n_S_dc_max_idx[j] = i

                # 利用f(t)=dc(t)+ac(t) 分别求dc和ac
                for j in range(6):

                    n_S_ac[j] = (an_S[j,int(val_loc[k+1])] - an_S[j,int(val_loc[k])])*(n_S_dc_max_idx[j] -val_loc[k])
                    n_S_ac[j] =  an_S[j,int(val_loc[k])] + n_S_ac[j]/ (val_loc[k+1] - val_loc[k])
                    n_S_ac[j] =  an_S[j,n_S_dc_max_idx[j]] - n_S_ac[j] # subtracting linear DC compoenents from raw

                    ac_dc_location[j,k,0] = n_S_ac[j]
                    ac_dc_location[j,k,1] = n_S_dc_max[j]
                    
                    if plot_flag:
                        # ax[j].hlines(n_S_dc_max[j],val_loc[k]-5,val_loc[k]+5,'r')               # for plot
                        # ax[j].hlines(n_S_dc_max[j]-n_S_ac[j],val_loc[k]-5,val_loc[k]+5,'r')     # for plot
                        # ax[j].vlines(val_loc[k],n_S_dc_max[j] - n_S_ac[j],n_S_dc_max[j],'r')    # for plot
                        line_color = 'green'
                        ax[j].vlines(val_loc[k],n_S_dc_max[j] - n_S_ac[j],n_S_dc_max[j],line_color)
                        ax[j].vlines(val_loc[k+1],n_S_dc_max[j] - n_S_ac[j],n_S_dc_max[j],line_color)
                        ax[j].hlines(n_S_dc_max[j] - n_S_ac[j],val_loc[k],val_loc[k+1],line_color)
                        ax[j].hlines(n_S_dc_max[j],val_loc[k],val_loc[k+1],line_color)
                        

                n_nume = n_S_ac[0] * n_S_dc_max[0] * n_S_ac[2] * n_S_dc_max[3]
                n_denom = n_S_ac[1] * n_S_dc_max[1] * n_S_ac[3] * n_S_dc_max[2]

                if (n_denom > 0 and n_i_ratio_count < len(val_loc) and n_nume != 0):
                    for j in range(6):
                        S_AC_DC[2*j,n_i_ratio_count] = round(n_S_ac[j],2)
                        S_AC_DC[2*j+1,n_i_ratio_count] = round(n_S_dc_max[j],2)
                    n_i_ratio_count += 1
                    
        if plot_flag:
            fig_test.savefig('peak_location' + '.pdf')  # for plot

        n_middle_idx = int(n_i_ratio_count/2)

        if plot_flag:
            for j in range(6):
                line_color = 'red'
                k = n_middle_idx
                ax[j].vlines(val_loc[k],ac_dc_location[j,k,0],ac_dc_location[j,k,0] - ac_dc_location[j,k,1],line_color)
                ax[j].vlines(val_loc[k+1],ac_dc_location[j,k,0],ac_dc_location[j,k,0] - ac_dc_location[j,k,1],line_color)
                ax[j].hlines(ac_dc_location[j,k,0] - ac_dc_location[j,k,1],val_loc[k],val_loc[k+1],line_color)
                ax[j].hlines(ac_dc_location[j,k,0],val_loc[k],val_loc[k+1],line_color)
        return S_AC_DC[:,n_middle_idx],ac_dc_location

    def TH_preprocess(self,data_in): # 温湿度数据预处理
        data_out = [data_in[0,0],data_in[0,5*9-1],np.mean(data_in[0,:]),
                    data_in[1,0],data_in[1,5*9-1],np.mean(data_in[1,:]),
                    data_in[2,0],data_in[2,5*9-1],np.mean(data_in[2,:]),
                    (data_in[0,5*3-1]-data_in[0,0])/3,(data_in[0,5*6-1]-data_in[0,5*3-1])/3,(data_in[0,5*9-1]-data_in[0,5*6-1])/3,
                    (data_in[1,5*3-1]-data_in[1,0])/3,(data_in[1,5*6-1]-data_in[1,5*3-1])/3,(data_in[1,5*9-1]-data_in[1,5*6-1])/3,
                    (data_in[2,5*3-1]-data_in[2,0])/3,(data_in[2,5*6-1]-data_in[2,5*3-1])/3,(data_in[2,5*9-1]-data_in[2,5*6-1])/3]
        return data_out

