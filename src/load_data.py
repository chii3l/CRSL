import os
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

import src.data_process_workingcopy as data_process


def _install_numpy_pickle_compat():
    try:
        import numpy.core as numpy_core
        sys.modules.setdefault("numpy._core", numpy_core)
        for submodule_name in ["multiarray", "numeric", "_multiarray_umath"]:
            legacy_name = f"numpy.core.{submodule_name}"
            modern_name = f"numpy._core.{submodule_name}"
            if legacy_name in sys.modules:
                sys.modules.setdefault(modern_name, sys.modules[legacy_name])
    except Exception:
        pass


class data_provider():
    def __init__(self):
        self.X_train_list = []
        self.X_test_list = []
        self.Y_train = []
        self.Y_test = []
        self.BG_Min_Max = []
        self.detail_bin = "Modal_"
        self.modal_enable_dic = {"modal_1": 1, "modal_2": 1, "modal_3": 1, "modal_4": 1, "modal_5": 0}
        self.ppg_enable_dic = {"ppg_1": 1, "ppg_2": 1, "ppg_3": 1, "ppg_4": 1, "ppg_5": 1, "ppg_6": 1}
        self.split_mode = "record"
        self.split_label = "record_test20_seed2"
        self.test_size = 0.2
        self.random_state = 2
        self.patient_column = 3
        self.patient_ids = None
        self.validation_group = None
        self.validation_group_name = None
        self.glucose_subwindow_label = ""
        self.glucose_subwindow_info = {}
        self.subject_counts = {}
        self.split_stats = {}
        self.dataset_name = "glucose"
        self.target_name = "BG"
        self.target_unit = "mmol/L"
        self.enable_error_grid = True

    def _record_split(self, n_samples, test_size, random_state):
        splitter = ShuffleSplit(n_splits=1, random_state=int(random_state), test_size=float(test_size))
        indices = np.arange(int(n_samples))
        return next(splitter.split(indices))

    def _patient_split(self, groups, test_size, random_state):
        groups = np.asarray(groups).astype(str)
        splitter = GroupShuffleSplit(n_splits=1, random_state=int(random_state), test_size=float(test_size))
        indices = np.arange(len(groups))
        return next(splitter.split(indices, groups=groups))

    def _balanced_patient_split(self, groups, y, test_size, random_state, n_trials=4096):
        groups = np.asarray(groups).astype(str)
        y = np.asarray(y, dtype=float).reshape(len(groups), -1)[:, 0]
        unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        n_samples = len(groups)
        n_groups = len(unique_groups)
        target_samples = max(1, int(round(float(test_size) * n_samples)))
        target_groups = max(1, int(round(float(test_size) * n_groups)))

        group_sum = np.bincount(inverse, weights=y, minlength=n_groups)
        group_sq_sum = np.bincount(inverse, weights=y * y, minlength=n_groups)
        overall_mean = float(np.mean(y))
        overall_std = float(np.std(y)) + 1e-8

        quantiles = np.linspace(0.0, 1.0, 6)
        bins = np.unique(np.quantile(y, quantiles))
        if len(bins) < 3:
            bins = np.linspace(float(np.min(y)), float(np.max(y)) + 1e-8, 6)
        bins[0] -= 1e-8
        bins[-1] += 1e-8
        y_bins = np.digitize(y, bins[1:-1], right=True)
        n_bins = len(bins) - 1
        group_hist = np.zeros((n_groups, n_bins), dtype=float)
        for idx in range(n_groups):
            group_hist[idx] = np.bincount(y_bins[inverse == idx], minlength=n_bins)
        overall_hist = np.bincount(y_bins, minlength=n_bins).astype(float)
        overall_hist = overall_hist / max(float(np.sum(overall_hist)), 1.0)

        def score(selected):
            selected = np.asarray(selected, dtype=int)
            test_count = int(np.sum(counts[selected]))
            if test_count <= 0 or test_count >= n_samples:
                return float("inf")
            test_group_count = len(selected)
            test_sum = float(np.sum(group_sum[selected]))
            test_sq_sum = float(np.sum(group_sq_sum[selected]))
            test_mean = test_sum / test_count
            test_var = max(test_sq_sum / test_count - test_mean * test_mean, 0.0)
            test_std = float(np.sqrt(test_var))
            test_hist = np.sum(group_hist[selected], axis=0)
            test_hist = test_hist / max(float(np.sum(test_hist)), 1.0)
            count_error = abs(test_count - target_samples) / max(float(target_samples), 1.0)
            subject_error = abs(test_group_count - target_groups) / max(float(target_groups), 1.0)
            mean_error = abs(test_mean - overall_mean) / overall_std
            std_error = abs(test_std - overall_std) / overall_std
            hist_error = float(np.sum(np.abs(test_hist - overall_hist)))
            max_share = float(np.max(counts[selected])) / max(float(test_count), 1.0)
            max_share_penalty = max(0.0, max_share - 0.15)
            return (
                5.0 * count_error
                + 0.30 * subject_error
                + 1.25 * mean_error
                + 0.75 * std_error
                + 0.75 * hist_error
                + 8.0 * max_share_penalty)

        rng = np.random.default_rng(int(random_state))
        best_selected = None
        best_score = float("inf")

        order_by_count = np.argsort(-counts)
        candidate_orders = [order_by_count, order_by_count[::-1]]
        for _ in range(max(int(n_trials), n_groups * 16)):
            candidate_orders.append(rng.permutation(n_groups))

        for order in candidate_orders:
            selected = []
            selected_count = 0
            for group_idx in order:
                if selected_count >= target_samples and selected:
                    break
                selected.append(int(group_idx))
                selected_count += int(counts[group_idx])
            if len(selected) > 1:
                without_last = selected[:-1]
                if abs(np.sum(counts[without_last]) - target_samples) < abs(selected_count - target_samples):
                    selected = without_last
            current_score = score(selected)
            if current_score < best_score:
                best_score = current_score
                best_selected = np.asarray(selected, dtype=int)

        test_groups = unique_groups[best_selected]
        test_mask = np.isin(groups, test_groups)
        indices = np.arange(n_samples)
        train_index = indices[~test_mask]
        test_index = indices[test_mask]
        self.split_stats = {
            "target_test_samples": target_samples,
            "actual_test_samples": int(len(test_index)),
            "target_test_subjects": target_groups,
            "actual_test_subjects": int(len(test_groups)),
            "max_test_subject_windows": int(np.max(counts[best_selected])),
            "max_test_subject_share": float(np.max(counts[best_selected]) / max(len(test_index), 1)),
            "score": float(best_score),
        }
        return train_index, test_index

    def _split_full_arrays(self, X_full_list, Y_full, data_raw, split_mode, test_size, random_state, patient_column):
        split_mode = str(split_mode).lower()
        if split_mode in ["record", "random", "row"]:
            train_index, test_index = self._record_split(len(Y_full), test_size, random_state)
            split_mode = "record"
            self.split_stats = {
                "target_test_samples": int(round(float(test_size) * len(Y_full))),
                "actual_test_samples": int(len(test_index)),
            }
        elif split_mode in ["patient", "subject", "patient_balanced", "subject_balanced"]:
            patient_ids = np.asarray(data_raw[:, int(patient_column)]).astype(str)
            train_index, test_index = self._balanced_patient_split(patient_ids, Y_full, test_size, random_state)
            split_mode = "patient_balanced"
        elif split_mode in ["patient_random", "subject_random"]:
            patient_ids = np.asarray(data_raw[:, int(patient_column)]).astype(str)
            train_index, test_index = self._patient_split(patient_ids, test_size, random_state)
            split_mode = "patient_random"
            self.split_stats = {
                "target_test_samples": int(round(float(test_size) * len(Y_full))),
                "actual_test_samples": int(len(test_index)),
            }
        else:
            raise ValueError("glucose split_mode must be one of: record, patient, patient_random")

        X_train_list = [x[train_index] for x in X_full_list]
        X_test_list = [x[test_index] for x in X_full_list]
        Y_train = Y_full[train_index]
        Y_test = Y_full[test_index]
        split_label = f"{split_mode}_test{int(float(test_size) * 100)}_seed{int(random_state)}"
        return X_train_list, X_test_list, Y_train, Y_test, split_mode, split_label, train_index, test_index

    def _restore_full_from_legacy_cache(self, X_train_list, X_test_list, Y_train, Y_test, n_samples):
        legacy_train_index, legacy_test_index = self._record_split(n_samples, 1 / 5, 2)
        if len(legacy_train_index) != len(Y_train) or len(legacy_test_index) != len(Y_test):
            raise ValueError(
                "Cached glucose split cannot be restored to full arrays. "
                "Run with --no-load-cache once to rebuild full cached arrays.")

        X_full_list = []
        for x_train, x_test in zip(X_train_list, X_test_list):
            x_full = np.empty((n_samples,) + x_train.shape[1:], dtype=x_train.dtype)
            x_full[legacy_train_index] = x_train
            x_full[legacy_test_index] = x_test
            X_full_list.append(x_full)

        Y_full = np.empty((n_samples,) + Y_train.shape[1:], dtype=Y_train.dtype)
        Y_full[legacy_train_index] = Y_train
        Y_full[legacy_test_index] = Y_test
        return X_full_list, Y_full

    def _build_full_arrays_from_raw(self, file_path):
        num_files = 1000
        data_modify_train = data_process.Data_process(file_path, num_files, plot=False)
        data_modify_train.load_data()
        data_modify_train.data_modify(enable_default=True)
        data_modify_train.initialize_data()
        data_modify_train.feature_extration(enable_smooth=False, enable_top_k=True)
        data_modify_train.data_scale()

        X_full_list = [
            data_modify_train.dataset_LED,
            np.concatenate((data_modify_train.dataset_Tem, data_modify_train.dataset_Hum), axis=1),
            np.concatenate((data_modify_train.dataset_LED_feature[:, :2], data_modify_train.dataset_p_info[:, :-1]), axis=1),
            data_modify_train.dataset_LED_feature[:, 2:].reshape(-1, 6, 8),
            data_modify_train.dataset_p_info[:, -1][:, None],
        ]
        Y_full = data_modify_train.dataset_BG.astype("float64")
        BG_Min_Max = [data_modify_train.BG_Min_Max[0][0], data_modify_train.BG_Min_Max[1][0]]
        return X_full_list, Y_full, BG_Min_Max, data_modify_train.data_raw, data_modify_train.BG_Min_Max

    def _load_full_arrays_from_cache(self, save_dir):
        _install_numpy_pickle_compat()
        BG_Min_Max = np.load(os.path.join(save_dir, "BG_Min_Max.npy"), allow_pickle=True)
        data_raw = np.load(os.path.join(save_dir, "data_raw.npy"), allow_pickle=True)

        full_paths = [os.path.join(save_dir, f"X_full_modal{i + 1}.npy") for i in range(5)]
        y_full_path = os.path.join(save_dir, "Y_full.npy")
        if all(os.path.exists(path) for path in full_paths) and os.path.exists(y_full_path):
            X_full_list = [np.load(path, allow_pickle=True) for path in full_paths]
            Y_full = np.load(y_full_path, allow_pickle=True)
        else:
            X_train_list = [np.load(os.path.join(save_dir, f"X_train_modal{i + 1}.npy"), allow_pickle=True) for i in range(5)]
            X_test_list = [np.load(os.path.join(save_dir, f"X_test_modal{i + 1}.npy"), allow_pickle=True) for i in range(5)]
            Y_train = np.load(os.path.join(save_dir, "Y_train.npy"), allow_pickle=True)
            Y_test = np.load(os.path.join(save_dir, "Y_test.npy"), allow_pickle=True)
            X_full_list, Y_full = self._restore_full_from_legacy_cache(
                X_train_list,
                X_test_list,
                Y_train,
                Y_test,
                len(data_raw))
        return X_full_list, Y_full, BG_Min_Max, data_raw

    def _save_cache(self, save_dir, X_full_list, Y_full, X_train_list, X_test_list, Y_train, Y_test, BG_Min_Max, data_raw):
        os.makedirs(save_dir, exist_ok=True)
        for i, data in enumerate(X_full_list):
            np.save(os.path.join(save_dir, f"X_full_modal{i + 1}.npy"), data, allow_pickle=True)
        np.save(os.path.join(save_dir, "Y_full.npy"), Y_full, allow_pickle=True)

        for i, data in enumerate(X_train_list):
            np.save(os.path.join(save_dir, f"X_train_modal{i + 1}.npy"), data, allow_pickle=True)
        for i, data in enumerate(X_test_list):
            np.save(os.path.join(save_dir, f"X_test_modal{i + 1}.npy"), data, allow_pickle=True)

        np.save(os.path.join(save_dir, "Y_train.npy"), Y_train, allow_pickle=True)
        np.save(os.path.join(save_dir, "Y_test.npy"), Y_test, allow_pickle=True)
        np.save(os.path.join(save_dir, "BG_Min_Max.npy"), BG_Min_Max, allow_pickle=True)
        np.save(os.path.join(save_dir, "data_raw.npy"), data_raw, allow_pickle=True)

    def _expand_modal_by_subwindows(self, x, starts, window_points, reference_steps):
        x = np.asarray(x)
        if x.ndim >= 3 and x.shape[-1] == int(reference_steps):
            sliced = np.stack(
                [x[..., start:start + window_points] for start in starts],
                axis=1)
            return sliced.reshape((x.shape[0] * len(starts),) + sliced.shape[2:])
        return np.repeat(x, len(starts), axis=0)

    def _apply_glucose_subwindows(
            self,
            X_list,
            Y,
            source_indices,
            patient_ids,
            window_sec,
            stride_sec,
            sampling_rate):
        window_sec = float(window_sec)
        if window_sec <= 0:
            return X_list, Y, np.asarray(source_indices).astype(str), np.asarray(patient_ids).astype(str)

        ppg = np.asarray(X_list[0])
        if ppg.ndim != 3:
            raise ValueError("Glucose subwindowing expects PPG with shape (N, C, T).")

        sampling_rate = float(sampling_rate)
        stride_sec = float(stride_sec)
        if sampling_rate <= 0 or stride_sec <= 0:
            raise ValueError("glucose_subwindow_sampling_rate and glucose_subwindow_stride_sec must be positive.")

        n_steps = int(ppg.shape[-1])
        window_points = int(round(window_sec * sampling_rate))
        stride_points = int(round(stride_sec * sampling_rate))
        if window_points <= 0 or stride_points <= 0:
            raise ValueError("Resolved glucose subwindow length and stride must be positive.")
        if window_points > n_steps:
            raise ValueError(
                f"Glucose subwindow length {window_points} exceeds PPG length {n_steps}.")

        starts = list(range(0, n_steps - window_points + 1, stride_points))
        if not starts:
            raise ValueError("Glucose subwindowing produced no starts.")

        expanded_X = [
            self._expand_modal_by_subwindows(x, starts, window_points, n_steps)
            for x in X_list
        ]
        expanded_Y = np.repeat(Y, len(starts), axis=0)
        expanded_source = np.repeat(np.asarray(source_indices).astype(str), len(starts), axis=0)
        expanded_patient = np.repeat(np.asarray(patient_ids).astype(str), len(starts), axis=0)

        self.glucose_subwindow_info = {
            "enabled": True,
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "sampling_rate": sampling_rate,
            "window_points": int(window_points),
            "stride_points": int(stride_points),
            "starts": [int(start) for start in starts],
            "starts_sec": [float(start) / sampling_rate for start in starts],
            "subwindows_per_record": int(len(starts)),
        }
        self.glucose_subwindow_label = (
            f"_subwin{window_sec:g}s_stride{stride_sec:g}s_len{window_points}"
        )
        return expanded_X, expanded_Y, expanded_source, expanded_patient

    def load_data(
            self,
            enable_load_data=True,
            file_path="data/glucose",
            split_mode="record",
            test_size=0.2,
            random_state=2,
            patient_column=3,
            subwindow_sec=0.0,
            subwindow_stride_sec=1.0,
            subwindow_sampling_rate=50.0):
        save_dir = os.fspath(file_path)

        if not enable_load_data:
            X_full_list, Y_full, BG_Min_Max, data_raw, raw_BG_Min_Max = self._build_full_arrays_from_raw(file_path)
        else:
            X_full_list, Y_full, BG_Min_Max, data_raw = self._load_full_arrays_from_cache(save_dir)
            raw_BG_Min_Max = BG_Min_Max

        X_train_list, X_test_list, Y_train, Y_test, split_mode, split_label, train_index, test_index = self._split_full_arrays(
            X_full_list,
            Y_full,
            data_raw,
            split_mode,
            test_size,
            random_state,
            patient_column)

        if not enable_load_data or not os.path.exists(os.path.join(save_dir, "Y_full.npy")):
            self._save_cache(save_dir, X_full_list, Y_full, X_train_list, X_test_list, Y_train, Y_test, raw_BG_Min_Max, data_raw)

        full_patient_ids = np.asarray(data_raw[:, int(patient_column)]).astype(str)
        train_patient_ids = full_patient_ids[train_index]
        test_patient_ids = full_patient_ids[test_index]
        train_source_ids = np.asarray(train_index).astype(str)
        test_source_ids = np.asarray(test_index).astype(str)

        X_train_list, Y_train, train_source_group, train_patient_group = self._apply_glucose_subwindows(
            X_train_list,
            Y_train,
            train_source_ids,
            train_patient_ids,
            subwindow_sec,
            subwindow_stride_sec,
            subwindow_sampling_rate)
        X_test_list, Y_test, _, _ = self._apply_glucose_subwindows(
            X_test_list,
            Y_test,
            test_source_ids,
            test_patient_ids,
            subwindow_sec,
            subwindow_stride_sec,
            subwindow_sampling_rate)

        self.X_train_list = X_train_list
        self.X_test_list = X_test_list
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.BG_Min_Max = BG_Min_Max
        self.data_raw = data_raw
        self.split_mode = split_mode
        self.split_label = split_label
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.patient_column = int(patient_column)
        self.patient_ids = full_patient_ids
        if split_mode in ["patient_balanced", "patient_random"]:
            self.validation_group = train_patient_group
            self.validation_group_name = "patient_id"
        elif self.glucose_subwindow_info.get("enabled", False):
            self.validation_group = train_source_group
            self.validation_group_name = "source_record_id"
        else:
            self.validation_group = None
            self.validation_group_name = None
        unique_ids, counts = np.unique(self.patient_ids, return_counts=True)
        self.subject_counts = dict(zip(unique_ids.tolist(), counts.astype(int).tolist()))

    def input_enable(
            self,
            modal_enable_dic={"modal_1": 1, "modal_2": 1, "modal_3": 1, "modal_4": 1, "modal_5": 0},
            ppg_enable_dic={"ppg_1": 1, "ppg_2": 1, "ppg_3": 1, "ppg_4": 1, "ppg_5": 1, "ppg_6": 1}):
        X_train_list_temp = self.X_train_list.copy()
        X_test_list_temp = self.X_test_list.copy()
        self.detail_bin = "Modal_"
        self.modal_enable_dic = modal_enable_dic.copy()
        self.ppg_enable_dic = ppg_enable_dic.copy()

        for key in modal_enable_dic:
            self.detail_bin += str(modal_enable_dic[key])

        for i in range(len(self.X_train_list)):
            if not modal_enable_dic[f"modal_{i + 1}"]:
                X_train_list_temp[i] = np.zeros(np.shape(self.X_train_list[i]))
                X_test_list_temp[i] = np.zeros(np.shape(self.X_test_list[i]))

        self.detail_bin += "_ppg_"
        for key in ppg_enable_dic:
            self.detail_bin += str(ppg_enable_dic[key])
        self.detail_bin += f"_glucose_{self.split_label}"
        if self.glucose_subwindow_label:
            self.detail_bin += self.glucose_subwindow_label

        index_mapping_2 = {
            "ppg_1": 0,
            "ppg_2": 1,
            "ppg_3": 2,
            "ppg_4": 3,
            "ppg_5": 4,
            "ppg_6": 5,
        }

        for key, value in ppg_enable_dic.items():
            if value == 0:
                X_train_list_temp[0][:, index_mapping_2[key]] = 0
                X_test_list_temp[0][:, index_mapping_2[key]] = 0
                X_train_list_temp[3][:, index_mapping_2[key]] = 0
                X_test_list_temp[3][:, index_mapping_2[key]] = 0

        self.X_train_list = X_train_list_temp
        self.X_test_list = X_test_list_temp

    def Input_check(self):
        Modal_name = ["PPG", "T&H", "Demo", "DF", "MI"]
        print("########### Input_check ###########")
        for i in range(len(self.X_train_list)):
            if np.all(self.X_train_list[i] == 0):
                print("Modal_%s Input:" % Modal_name[i], 0)
            else:
                print("Modal_%s Input:" % Modal_name[i], 1)

        for i in range(self.X_train_list[0].shape[1]):
            if np.all(self.X_train_list[0][:, i] == 0):
                print("PPG_%s Input:" % i, 0)
            else:
                print("PPG_%s Input:" % i, 1)
        print("#####################################\n")

    def Shape_check(self):
        Modal_name = ["PPG", "T&H", "Demo", "DF", "MI"]
        print("########### Shape_check ############")
        for i in range(len(self.X_train_list)):
            print("Modal_%s Shape:" % Modal_name[i], self.X_train_list[i].shape[1:])
        print("#####################################\n")
