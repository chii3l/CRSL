import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from openpyxl import Workbook, load_workbook
import src.error_grid_plot as error_grid_plot
from src.Model import Traning_and_Evaluation


def _excel_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and np.isinf(value):
        return 'inf' if value > 0 else '-inf'
    return value


def _replace_dataframe_sheet(workbook, sheet_name, dataframe):
    if sheet_name in workbook.sheetnames:
        del workbook[sheet_name]
    worksheet = workbook.create_sheet(title=sheet_name)
    worksheet.append([str(column) for column in dataframe.columns])
    for row in dataframe.itertuples(index=False, name=None):
        worksheet.append([_excel_value(value) for value in row])


def _sanitize_loss_history(dataframe):
    """Return a numeric-only history frame that is safe to serialize."""
    cleaned = pd.DataFrame(index=dataframe.index)
    invalid_counts = {}

    for column in dataframe.columns:
        if column == 'epoch':
            cleaned[column] = np.arange(1, len(dataframe) + 1, dtype=np.int64)
            continue

        values = []
        invalid_count = 0
        for value in dataframe[column].tolist():
            if isinstance(value, np.generic):
                value = value.item()
            elif hasattr(value, 'item') and callable(value.item):
                try:
                    value = value.item()
                except (TypeError, ValueError, RuntimeError):
                    value = np.nan

            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                numeric_value = np.nan

            if not np.isfinite(numeric_value):
                numeric_value = np.nan
                invalid_count += 1
            values.append(numeric_value)

        cleaned[column] = np.asarray(values, dtype=np.float64)
        if invalid_count:
            invalid_counts[column] = invalid_count

    return cleaned, invalid_counts


def _write_loss_history(dataframe, output_path):
    cleaned, invalid_counts = _sanitize_loss_history(dataframe)
    if invalid_counts:
        details = ', '.join(
            f'{column}={count}' for column, count in invalid_counts.items())
        print(
            'Warning: replaced non-finite or non-scalar loss-history values '
            f'with NaN before CSV export: {details}')

    temporary_path = f'{output_path}.tmp'
    try:
        cleaned.to_csv(
            temporary_path,
            index=False,
            float_format='%.12g')
        os.replace(temporary_path, output_path)
    except Exception as exc:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        print(
            'Warning: loss-history CSV export failed after training and was '
            f'skipped: {type(exc).__name__}: {exc}')


class Training_Process():
    def __init__(self, parameters_dic, dataset_dic):
        self.parameters_dic = parameters_dic
        self.dataset_dic = dataset_dic
        self.validation_group = self.dataset_dic.get("validation_group", None)
        self.validation_group_name = self.dataset_dic.get("validation_group_name", "group")
        if self.validation_group is not None:
            self.parameters_dic["validation_split_mode"] = f"group:{self.validation_group_name}"
        else:
            self.parameters_dic["validation_split_mode"] = "random"
        
        self.__directory_init()
        self.__result_list_init()
        self.index_splitter()

    def __directory_init(self):
        self.directory = os.getcwd()
        self.filename = {
                            '1st_level': datetime.now().strftime("%Y-%m-%d"),
                            '2nd_level': os.path.join(self.parameters_dic['detail'] + '_' + datetime.now().strftime("%H-%M-%S"))
                        }
        
        result_dir = os.path.join(self.directory, 'outputs', self.filename['1st_level'])
        sub_dir = os.path.join(result_dir, self.filename['2nd_level'])
        
        # Create necessary directories if they don't exist
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        self.save_path = sub_dir
        
        # Create subdirectories for model, training history, and error grid
        model_dir = os.path.join(self.save_path, 'Model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        train_hist_dir = os.path.join(self.save_path, 'Train_hist')
        if not os.path.exists(train_hist_dir):
            os.makedirs(train_hist_dir)

        error_grid_dir = os.path.join(self.save_path, 'Error_grid')
        if not os.path.exists(error_grid_dir):
            os.makedirs(error_grid_dir)

        spectral_output_dir = os.path.join(self.save_path, 'Spectral_output')
        if not os.path.exists(spectral_output_dir):
            os.makedirs(spectral_output_dir)
            
        # Create initial file in Excel format
        model_output_path = os.path.join(self.save_path, 'Model_output.xlsx')
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = 'Initial'
        worksheet.append(['Initial'])
        worksheet.append(['This is the initial sheet'])
        workbook.save(model_output_path)

        # Save parameters_dic to a text file
        params_file_path = os.path.join(self.save_path, 'parameters.txt')
        with open(params_file_path, 'w') as f:
            for key, value in self.parameters_dic.items():
                f.write(f"{key}: {value}\n")
                        
        print("Initialization complete. Files and directories created successfully.") 
        
    def __result_list_init(self):
        self.id_row = np.zeros((self.parameters_dic['n_split']+2,1,3))
        self.r2_RMSE_MAE_MAPE = np.zeros((self.parameters_dic['n_split']+2,4,3))
        self.error_percentage = np.zeros((self.parameters_dic['n_split']+2,5,3))
        self.zone_percentage = np.zeros((self.parameters_dic['n_split']+2,5,3))
        
    def index_splitter(self):
        X = np.arange(len(self.dataset_dic['X_train'][0])) # 数据编号
        validation_size = float(self.parameters_dic.get("validation_size", 0.25))
        if self.validation_group is not None:
            groups = np.asarray(self.validation_group).astype(str)
            if len(groups) != len(X):
                raise ValueError(
                    "validation_group length must match the number of training samples: "
                    f"{len(groups)} vs {len(X)}")
            if len(np.unique(groups)) < 2:
                print(
                    "Warning: validation_group has fewer than 2 unique groups; "
                    "falling back to random validation split.")
                ss = ShuffleSplit(
                    n_splits=self.parameters_dic['n_split'],
                    random_state=2,
                    test_size=validation_size)
                self.splitter = ss.split(X)
                self.parameters_dic["validation_split_mode"] = "random_fallback"
            else:
                ss = GroupShuffleSplit(
                    n_splits=self.parameters_dic['n_split'],
                    random_state=2,
                    test_size=validation_size)
                self.splitter = ss.split(X, groups=groups)
                print(
                    f"Validation split uses disjoint {self.validation_group_name} groups "
                    f"(n={len(np.unique(groups))}).")
        else:
            ss = ShuffleSplit(
                n_splits=self.parameters_dic['n_split'],
                random_state=2,
                test_size=validation_size)
            self.splitter = ss.split(X)
        
    def train_start(self):
        self.__n_round = 0
        for train_index, validation_index in self.splitter:
            self.parameters_dic["round_seed"] = (
                int(self.parameters_dic.get("training_seed", 2026)) + self.__n_round)
            train_X = [self.dataset_dic['X_train'][index][train_index] for index in range(len(self.dataset_dic['X_train']))]
            train_y = self.dataset_dic['Y_train'][train_index]
            validation_X = [self.dataset_dic['X_train'][index][validation_index] for index in range(len(self.dataset_dic['X_train']))]
            validation_y = self.dataset_dic['Y_train'][validation_index]
            test_X = [self.dataset_dic['X_test'][index] for index in range(len(self.dataset_dic['X_test']))]
            test_y = self.dataset_dic['Y_test']
            
            self.__TV = Traning_and_Evaluation(train_X, train_y, validation_X, validation_y, test_X, test_y)
            self.__TV.Train(self.parameters_dic)
            
            self.__TV.model_save(os.path.join(self.save_path, 'Model', '%s.pth'%str(self.__n_round)))
            
            self.__Train_hist_plot()
            
            self.__show_result(train_X, train_y, validation_X, validation_y, test_X, test_y)
                
            self.__n_round += 1
            
        self.__save_result_to_csv()
    
    
    def __model_save(self):
        # 获取模型摘要
        model_summary = self.__TV.get_model_summary()

        # 保存摘要到文件中
        with open(os.path.join(self.save_path, 'Model', '%s.txt'%str(self.__n_round)), 'w') as f:
            f.write(model_summary)

        return
            
    def __show_result(self,train_X, train_y, validation_X, validation_y, test_X, test_y):
        BG_min = self.dataset_dic['BG_Min_Max'][0]
        BG_max = self.dataset_dic['BG_Min_Max'][1]

        self.__BG_train_y_pre = self.__TV.model_eval(train_X, train_y, path = os.path.join(self.save_path, 'Model', '%s.pth'%str(self.__n_round)))
        self.__BG_validation_y_pre = self.__TV.model_eval(validation_X, validation_y, path = os.path.join(self.save_path, 'Model', '%s.pth'%str(self.__n_round)))
        self.__BG_test_y_pre = self.__TV.model_eval(test_X, test_y, path = os.path.join(self.save_path, 'Model', '%s.pth'%str(self.__n_round)))
                    
        self.__BG_predict_train = self.__scale_reverse(self.__BG_train_y_pre,BG_max,BG_min)
        self.__BG_predict_validation = self.__scale_reverse(self.__BG_validation_y_pre,BG_max,BG_min)
        self.__BG_predict_test = self.__scale_reverse(self.__BG_test_y_pre,BG_max,BG_min)
        
        self.__BG_true_train = self.__scale_reverse(train_y,BG_max,BG_min)
        self.__BG_true_validation = self.__scale_reverse(validation_y,BG_max,BG_min)
        self.__BG_true_test = self.__scale_reverse(test_y,BG_max,BG_min)

        self.__save_spectral_output(test_X, test_y)
        
        self.__save_result()
        
        if self.parameters_dic.get('enable_error_grid', True):
            self.__grid_plot()
        else:
            self.__regression_plot()
        
        self.__result_summary()
        
        return

    def __save_spectral_output(self, test_X, test_y):
        model_name = str(self.parameters_dic.get('model_name', 0)).lower()
        if model_name not in ['physics_spectral', 'spectral_physics', 'physics']:
            return

        model_path = os.path.join(self.save_path, 'Model', '%s.pth'%str(self.__n_round))
        max_samples = self.parameters_dic.get('spectral_export_samples', 16)
        batch_size = self.parameters_dic.get('spectral_export_batch_size', 16)
        aux_outputs = self.__TV.model_aux_eval(
            test_X,
            test_y,
            path=model_path,
            batch_size=batch_size,
            max_samples=max_samples)

        if aux_outputs is None:
            return

        BG_min = self.dataset_dic['BG_Min_Max'][0]
        BG_max = self.dataset_dic['BG_Min_Max'][1]
        bg_pred = self.__scale_reverse(aux_outputs['bg_pred'], BG_max, BG_min)
        bg_true = self.__scale_reverse(aux_outputs['bg_true'], BG_max, BG_min)
        target_names = self.parameters_dic.get(
            'target_names',
            self.dataset_dic.get('target_names', [self.parameters_dic.get('target_name', 'BG')]))
        target_units = self.parameters_dic.get(
            'target_units',
            self.dataset_dic.get('target_units', [self.parameters_dic.get('target_unit', 'mmol/L')]))

        save_path = os.path.join(self.save_path, 'Spectral_output', f'round_{self.__n_round}_samples.npz')
        payload = {
            'spectrum': aux_outputs['spectrum'],
            'y_hat': aux_outputs['y_hat'],
            'y_real': aux_outputs['y_real'],
            'H': aux_outputs['H'],
            'wavelengths': aux_outputs['wavelengths'],
            'bg_pred': bg_pred,
            'bg_true': bg_true,
            'target_pred': bg_pred,
            'target_true': bg_true,
            'target_name': np.asarray(target_names),
            'target_unit': np.asarray(target_units),
            'bg_pred_norm': aux_outputs['bg_pred'],
            'bg_true_norm': aux_outputs['bg_true']
        }
        if 'tokens' in aux_outputs:
            payload['tokens'] = aux_outputs['tokens']
        if 'gate_weights' in aux_outputs:
            payload['gate_weights'] = aux_outputs['gate_weights']
        if 'spectral_feature_gate' in aux_outputs:
            payload['spectral_feature_gate'] = aux_outputs['spectral_feature_gate']
        if 'spectral_basis' in aux_outputs:
            payload['spectral_basis'] = aux_outputs['spectral_basis']
        for key in [
                'clean_y_hat',
                'baseline_hat',
                'noise_hat',
                'background_spectrum',
                'target_spectrum',
                'target_glucose_spectrum',
                'target_spo2_spectrum',
                'primary_target_spectrum',
                'background_coefficients',
                'target_glucose_coefficients',
                'target_spo2_coefficients',
                'primary_target_coefficients',
                'glucose_component_pred',
                'spo2_component_pred',
                'primary_target_component_pred',
                'context_target',
                'background_context_pred',
                'target_context_pred',
                'background_context_film',
                'common_target_pred',
                'target_spectral_contribution',
                'target_spectral_gates',
                'background_spectral_basis',
                'target_spectral_basis',
                'target_glucose_spectral_basis',
                'target_spo2_spectral_basis']:
            if key in aux_outputs:
                payload[key] = aux_outputs[key]
        np.savez_compressed(save_path, **payload)
        print(f"Spectral outputs saved to {save_path}")
    
    def __save_result(self):
        # 创建 DataFrame，处理长度不等的数组
        max_length = max(len(self.__BG_predict_train), len(self.__BG_true_train), len(self.__BG_predict_validation), len(self.__BG_true_validation), len(self.__BG_predict_test), len(self.__BG_true_test))
        target_names = self.parameters_dic.get(
            'target_names',
            self.dataset_dic.get('target_names', [self.parameters_dic.get('target_name', 'BG')]))
        pred_train = np.asarray(self.__BG_predict_train)
        true_train = np.asarray(self.__BG_true_train)
        pred_validation = np.asarray(self.__BG_predict_validation)
        true_validation = np.asarray(self.__BG_true_validation)
        pred_test = np.asarray(self.__BG_predict_test)
        true_test = np.asarray(self.__BG_true_test)
        if pred_train.ndim == 1:
            pred_train = pred_train.reshape(-1, 1)
            true_train = true_train.reshape(-1, 1)
            pred_validation = pred_validation.reshape(-1, 1)
            true_validation = true_validation.reshape(-1, 1)
            pred_test = pred_test.reshape(-1, 1)
            true_test = true_test.reshape(-1, 1)

        data_dict = {}
        for idx in range(pred_train.shape[1]):
            target_name = target_names[idx] if idx < len(target_names) else f"target{idx}"
            data_dict[f'{target_name}_train_pre'] = pd.Series(pred_train[:, idx]).reindex(range(max_length))
            data_dict[f'{target_name}_train_ref'] = pd.Series(true_train[:, idx]).reindex(range(max_length))
            data_dict[f'{target_name}_validation_pre'] = pd.Series(pred_validation[:, idx]).reindex(range(max_length))
            data_dict[f'{target_name}_validation_ref'] = pd.Series(true_validation[:, idx]).reindex(range(max_length))
            data_dict[f'{target_name}_test_pre'] = pd.Series(pred_test[:, idx]).reindex(range(max_length))
            data_dict[f'{target_name}_test_ref'] = pd.Series(true_test[:, idx]).reindex(range(max_length))
        
        df = pd.DataFrame(data_dict)

        excel_path = os.path.join(self.save_path, 'Model_output.xlsx')
        sheet_name = f'Round_{self.__n_round}'
        
        # 检查文件是否存在
        book = load_workbook(excel_path)
        # 删除同名的 sheet
        _replace_dataframe_sheet(book, sheet_name, df)
        book.save(excel_path)
            
        print(f'数据已保存到 {excel_path} 文件中')
        
    def __grid_plot(self):
        self.parameters_dic["enable_PEG_summary_list"]
        train_true = self.__primary_target(self.__BG_true_train)
        train_pred = self.__primary_target(self.__BG_predict_train)
        val_true = self.__primary_target(self.__BG_true_validation)
        val_pred = self.__primary_target(self.__BG_predict_validation)
        test_true = self.__primary_target(self.__BG_true_test)
        test_pred = self.__primary_target(self.__BG_predict_test)
        
        self.__zone_error_rate_train = np.array(error_grid_plot.error_rate_count(train_true,train_pred))[[9,8,7,6,2]]/len(train_true)*100
        self.__zone_error_rate_validation = np.array(error_grid_plot.error_rate_count(val_true,val_pred))[[9,8,7,6,2]]/len(val_true)*100
        self.__zone_error_rate_test = np.array(error_grid_plot.error_rate_count(test_true,test_pred))[[9,8,7,6,2]]/len(test_true)*100
        
        fig_concensus = plt.figure(figsize=(20, 12))

        self.__zone_PEG_train, self.__r2_RMSE_MAE_MAPE_train = self.plot_error_grid(train_true, train_pred, 'Training Set', 1, self.parameters_dic['enable_PEG_20_40'], self.parameters_dic["enable_PEG_summary_list"][0])
        self.__zone_PEG_validation, self.__r2_RMSE_MAE_MAPE_validation = self.plot_error_grid(val_true, val_pred, 'Validation Set', 2, self.parameters_dic['enable_PEG_20_40'], self.parameters_dic["enable_PEG_summary_list"][1])
        self.__zone_PEG_test, self.__r2_RMSE_MAE_MAPE_test = self.plot_error_grid(test_true, test_pred, 'Testing Set Recording-level', 3, self.parameters_dic['enable_PEG_20_40'], self.parameters_dic["enable_PEG_summary_list"][2])
        
        # 保存图像
        fig_concensus.savefig(os.path.join(self.save_path, 'Error_grid', '%s.png'%str(self.__n_round)))
        fig_concensus.clear()
        return

    def __compute_regression_metrics(self, target_true, target_predict):
        try:
            R2 = r2_score(target_true, target_predict)
            RMSE = np.sqrt(mean_squared_error(target_true, target_predict))
            MAE = mean_absolute_error(target_true, target_predict)
            MAPE = mean_absolute_percentage_error(target_true, target_predict)
        except Exception:
            R2 = 0
            RMSE = 0
            MAE = 0
            MAPE = 0
        return np.array([R2, RMSE, MAE, MAPE])

    def __compute_error_percentages(self, target_true, target_predict):
        target_true = np.asarray(target_true).reshape(-1)
        target_predict = np.asarray(target_predict).reshape(-1)
        percentage_error = np.abs(target_predict - target_true) / np.maximum(np.abs(target_true), 1e-8) * 100
        return np.asarray([
            np.mean(percentage_error < 5) * 100,
            np.mean(percentage_error < 10) * 100,
            np.mean(percentage_error < 15) * 100,
            np.mean(percentage_error < 20) * 100,
            np.mean(percentage_error < 40) * 100,
        ])

    def __regression_subplot(self, target_true, target_predict, title, subplot_index):
        target_name = self.parameters_dic.get('target_name', 'Target')
        target_unit = self.parameters_dic.get('target_unit', '')
        error_percentage = self.__compute_error_percentages(target_true, target_predict)
        metrics = self.__compute_regression_metrics(target_true, target_predict)

        plt.subplot(2, 3, subplot_index)
        plt.scatter(target_true, target_predict, s=12, alpha=0.6)
        low = min(np.min(target_true), np.min(target_predict))
        high = max(np.max(target_true), np.max(target_predict))
        margin = (high - low) * 0.05 if high > low else 1.0
        plt.plot([low - margin, high + margin], [low - margin, high + margin], 'k--', linewidth=1)
        plt.xlabel(f"Reference {target_name} ({target_unit})")
        plt.ylabel(f"Predicted {target_name} ({target_unit})")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, subplot_index + 3)
        plt.axis('off')
        cell_text = [
            ['len_data', len(target_true)],
            ['<5%', error_percentage[0]],
            ['<10%', error_percentage[1]],
            ['<15%', error_percentage[2]],
            ['<20%', error_percentage[3]],
            ['<40%', error_percentage[4]],
            ['R2', metrics[0]],
            ['RMSE', metrics[1]],
            ['MAE', metrics[2]],
            ['MAPE', metrics[3]]
        ]
        table = plt.table(cellText=cell_text, colLabels=('Metric', 'Value'), loc='center', cellLoc='center', colLoc='center')
        table.scale(1, 2)
        return error_percentage, metrics

    def __regression_plot(self):
        fig = plt.figure(figsize=(20, 12))
        self.__zone_error_rate_train, self.__r2_RMSE_MAE_MAPE_train = self.__regression_subplot(
            self.__BG_true_train, self.__BG_predict_train, 'Training Set', 1)
        self.__zone_error_rate_validation, self.__r2_RMSE_MAE_MAPE_validation = self.__regression_subplot(
            self.__BG_true_validation, self.__BG_predict_validation, 'Validation Set', 2)
        self.__zone_error_rate_test, self.__r2_RMSE_MAE_MAPE_test = self.__regression_subplot(
            self.__BG_true_test, self.__BG_predict_test, 'Testing Set Recording-level', 3)
        self.__zone_PEG_train = np.full(5, np.nan)
        self.__zone_PEG_validation = np.full(5, np.nan)
        self.__zone_PEG_test = np.full(5, np.nan)
        fig.savefig(os.path.join(self.save_path, 'Error_grid', '%s.png'%str(self.__n_round)))
        plt.close(fig)
        return
    
    def __Train_hist_plot(self):
        epochs = range(1, len(self.__TV.train_losses) + 1)
        has_physics_losses = (
            hasattr(self.__TV, 'train_bg_losses') and
            hasattr(self.__TV, 'val_bg_losses') and
            len(self.__TV.train_bg_losses) == len(self.__TV.train_losses)
        )
        has_decorr_losses = (
            has_physics_losses and
            hasattr(self.__TV, 'train_decorr_losses') and
            hasattr(self.__TV, 'val_decorr_losses') and
            len(self.__TV.train_decorr_losses) == len(self.__TV.train_losses)
        )

        if has_physics_losses:
            fig_Train_hist = plt.figure(figsize=(14, 8))
            axes_shape = (2, 3)
        else:
            fig_Train_hist = plt.figure(figsize=(10, 4))
            axes_shape = (1, 2)

        plt.subplot(*axes_shape, 1)
        plt.plot(epochs, self.__TV.train_losses, label="Training Loss")
        plt.plot(epochs, self.__TV.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.legend()
        plt.grid(True)

        if has_physics_losses:
            plt.subplot(*axes_shape, 2)
            target_name = self.parameters_dic.get('target_name', 'BG')
            plt.plot(epochs, self.__TV.train_bg_losses, label=f"Training {target_name} Loss")
            plt.plot(epochs, self.__TV.val_bg_losses, label=f"Validation {target_name} Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"{target_name} Prediction Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(*axes_shape, 3)
            plt.plot(epochs, self.__TV.train_obs_losses, label="Training Weighted Obs Loss")
            plt.plot(epochs, self.__TV.val_obs_losses, label="Validation Weighted Obs Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Weighted Observation Loss")
            plt.legend()
            plt.grid(True)

            plt.subplot(*axes_shape, 4)
            plt.plot(epochs, self.__TV.train_smooth_losses, label="Training Weighted Smooth Loss")
            plt.plot(epochs, self.__TV.val_smooth_losses, label="Validation Weighted Smooth Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Weighted Smoothness Loss")
            plt.legend()
            plt.grid(True)

            if has_decorr_losses:
                plt.subplot(*axes_shape, 5)
                plt.plot(epochs, self.__TV.train_decorr_losses, label="Training Weighted Decorr Loss")
                plt.plot(epochs, self.__TV.val_decorr_losses, label="Validation Weighted Decorr Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title("Weighted Decorrelation Loss")
                plt.legend()
                plt.grid(True)
                lr_subplot_index = 6
            else:
                lr_subplot_index = 5
        else:
            lr_subplot_index = 2

        plt.subplot(*axes_shape, lr_subplot_index)
        plt.plot(epochs, self.__TV.lrs, label="Learning Rate", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")
        plt.legend()
        plt.grid(True)

        if has_physics_losses:
            loss_history = pd.DataFrame({
                'epoch': list(epochs),
                'train_total_loss': self.__TV.train_losses,
                'val_total_loss': self.__TV.val_losses,
                'train_bg_loss': self.__TV.train_bg_losses,
                'val_bg_loss': self.__TV.val_bg_losses,
                'train_obs_loss': self.__TV.train_obs_losses,
                'val_obs_loss': self.__TV.val_obs_losses,
                'train_smooth_loss': self.__TV.train_smooth_losses,
                'val_smooth_loss': self.__TV.val_smooth_losses,
                'learning_rate': self.__TV.lrs
            })
            if has_decorr_losses:
                loss_history['train_decorr_loss'] = self.__TV.train_decorr_losses
                loss_history['val_decorr_loss'] = self.__TV.val_decorr_losses
            if hasattr(self.__TV, 'train_obs_raw_losses'):
                loss_history['train_obs_raw_loss'] = self.__TV.train_obs_raw_losses
                loss_history['val_obs_raw_loss'] = self.__TV.val_obs_raw_losses
                loss_history['train_smooth_raw_loss'] = self.__TV.train_smooth_raw_losses
                loss_history['val_smooth_raw_loss'] = self.__TV.val_smooth_raw_losses
            if hasattr(self.__TV, 'train_smooth_abs_losses'):
                loss_history['train_smooth_relative_loss'] = self.__TV.train_smooth_raw_losses
                loss_history['val_smooth_relative_loss'] = self.__TV.val_smooth_raw_losses
                loss_history['train_smooth_abs_loss'] = self.__TV.train_smooth_abs_losses
                loss_history['val_smooth_abs_loss'] = self.__TV.val_smooth_abs_losses
                loss_history['train_spectrum_rms'] = self.__TV.train_spectrum_rms_values
                loss_history['val_spectrum_rms'] = self.__TV.val_spectrum_rms_values
                loss_history['train_background_smooth_loss'] = self.__TV.train_background_smooth_losses
                loss_history['val_background_smooth_loss'] = self.__TV.val_background_smooth_losses
                loss_history['train_target_smooth_loss'] = self.__TV.train_target_smooth_losses
                loss_history['val_target_smooth_loss'] = self.__TV.val_target_smooth_losses
                loss_history['train_basis_smooth_loss'] = self.__TV.train_basis_smooth_losses
                loss_history['val_basis_smooth_loss'] = self.__TV.val_basis_smooth_losses
            if has_decorr_losses and hasattr(self.__TV, 'train_decorr_raw_losses'):
                loss_history['train_decorr_raw_loss'] = self.__TV.train_decorr_raw_losses
                loss_history['val_decorr_raw_loss'] = self.__TV.val_decorr_raw_losses
            if hasattr(self.__TV, 'train_bg_context_losses'):
                loss_history['train_bg_context_loss'] = self.__TV.train_bg_context_losses
                loss_history['val_bg_context_loss'] = self.__TV.val_bg_context_losses
                loss_history['train_target_context_loss'] = self.__TV.train_target_context_losses
                loss_history['val_target_context_loss'] = self.__TV.val_target_context_losses
            if hasattr(self.__TV, 'train_bg_context_raw_losses'):
                loss_history['train_bg_context_raw_loss'] = self.__TV.train_bg_context_raw_losses
                loss_history['val_bg_context_raw_loss'] = self.__TV.val_bg_context_raw_losses
                loss_history['train_target_context_raw_loss'] = self.__TV.train_target_context_raw_losses
                loss_history['val_target_context_raw_loss'] = self.__TV.val_target_context_raw_losses
            if hasattr(self.__TV, 'train_glucose_component_losses'):
                loss_history['train_glucose_component_loss'] = self.__TV.train_glucose_component_losses
                loss_history['val_glucose_component_loss'] = self.__TV.val_glucose_component_losses
                loss_history['train_spo2_component_loss'] = self.__TV.train_spo2_component_losses
                loss_history['val_spo2_component_loss'] = self.__TV.val_spo2_component_losses
                if self.parameters_dic.get('enable_single_target_residual', False):
                    loss_history['train_primary_target_component_loss'] = (
                        self.__TV.train_glucose_component_losses)
                    loss_history['val_primary_target_component_loss'] = (
                        self.__TV.val_glucose_component_losses)
            if hasattr(self.__TV, 'train_glucose_component_raw_losses'):
                loss_history['train_glucose_component_raw_loss'] = self.__TV.train_glucose_component_raw_losses
                loss_history['val_glucose_component_raw_loss'] = self.__TV.val_glucose_component_raw_losses
                loss_history['train_spo2_component_raw_loss'] = self.__TV.train_spo2_component_raw_losses
                loss_history['val_spo2_component_raw_loss'] = self.__TV.val_spo2_component_raw_losses
                if self.parameters_dic.get('enable_single_target_residual', False):
                    loss_history['train_primary_target_component_raw_loss'] = (
                        self.__TV.train_glucose_component_raw_losses)
                    loss_history['val_primary_target_component_raw_loss'] = (
                        self.__TV.val_glucose_component_raw_losses)
        else:
            loss_history = pd.DataFrame({
                'epoch': list(epochs),
                'train_total_loss': self.__TV.train_losses,
                'val_total_loss': self.__TV.val_losses,
                'learning_rate': self.__TV.lrs
            })
        _write_loss_history(
            loss_history,
            os.path.join(
                self.save_path,
                'Train_hist',
                f'{self.__n_round}_loss_history.csv'))

        plt.tight_layout()
        fig_Train_hist.savefig(os.path.join(self.save_path, 'Train_hist', '%s.png'%str(self.__n_round)))
        plt.close(fig_Train_hist)

        has_smooth_diagnostics = (
            has_physics_losses and
            hasattr(self.__TV, 'train_smooth_abs_losses') and
            any(value > 0 for value in self.__TV.train_spectrum_rms_values)
        )
        if has_smooth_diagnostics:
            fig_smooth, axes = plt.subplots(2, 2, figsize=(14, 8))
            diagnostic_series = [
                (axes[0, 0], self.__TV.train_smooth_raw_losses,
                 self.__TV.val_smooth_raw_losses,
                 'Relative Spectral Curvature', 'Relative curvature'),
                (axes[0, 1], self.__TV.train_smooth_abs_losses,
                 self.__TV.val_smooth_abs_losses,
                 'Absolute First Difference', 'Absolute smoothness'),
                (axes[1, 0], self.__TV.train_spectrum_rms_values,
                 self.__TV.val_spectrum_rms_values,
                 'Spectrum RMS', 'RMS amplitude'),
            ]
            for axis, train_values, val_values, title, ylabel in diagnostic_series:
                axis.plot(epochs, train_values, label='Training')
                axis.plot(epochs, val_values, label='Validation')
                axis.set_xlabel('Epochs')
                axis.set_ylabel(ylabel)
                axis.set_title(title)
                axis.grid(True)
                axis.legend()

            component_axis = axes[1, 1]
            component_axis.plot(
                epochs, self.__TV.val_background_smooth_losses,
                label='Background weighted')
            component_axis.plot(
                epochs, self.__TV.val_target_smooth_losses,
                label='Targets weighted')
            component_axis.plot(
                epochs, self.__TV.val_basis_smooth_losses,
                label='Bases weighted')
            component_axis.set_xlabel('Epochs')
            component_axis.set_ylabel('Weighted loss')
            component_axis.set_title('Validation Component Smoothness')
            component_axis.grid(True)
            component_axis.legend()
            fig_smooth.tight_layout()
            fig_smooth.savefig(os.path.join(
                self.save_path,
                'Train_hist',
                f'{self.__n_round}_smooth_diagnostics.png'))
            plt.close(fig_smooth)
        
    def __primary_target(self, data):
        data = np.asarray(data)
        if data.ndim == 2:
            return data[:, 0]
        return data.reshape(-1)

    def __scale_reverse(self,data,max,min):
        data = np.asarray(data)
        data_out = data * (np.asarray(max) - np.asarray(min)) + np.asarray(min)
        if data_out.ndim == 2 and data_out.shape[1] == 1:
            return data_out.reshape(-1,)
        return data_out

    def plot_error_grid(self, BG_true, BG_predict, title, subplot_index, enable_PEG_20_40, enable_PEG_summary):

        plt.subplot(2, 3, subplot_index)
        zone_error_rate = np.array(error_grid_plot.error_rate_count(BG_true, BG_predict))[[9, 8, 7, 6, 2]] / len(BG_true) * 100
        zone_PEG = error_grid_plot.consensus_error_grid(BG_true, BG_predict, title, 'mmol/L', enable_PEG_20_40, enable_PEG_summary=enable_PEG_summary)
        zone_PEG = np.array([zone_PEG[0], zone_PEG[1], zone_PEG[2], zone_PEG[3], zone_PEG[4]]) / len(BG_true) * 100
        try:
            R2 = r2_score(BG_true, BG_predict)
            RMSE = np.sqrt(mean_squared_error(BG_true, BG_predict))
            MAE = mean_absolute_error(BG_true, BG_predict)
            MAPE = mean_absolute_percentage_error(BG_true, BG_predict)
        except:
            R2 = 0
            RMSE = 0
            MAE = 0
            MAPE = 0

        plt.subplot(2, 3, subplot_index + 3)
        plt.axis('off')
        columns = ('Metric', 'Value')
        cell_text = [
            ['len_data', len(BG_true)],
            ['<5%', zone_error_rate[0]],
            ['<10%', zone_error_rate[1]],
            ['<15%', zone_error_rate[2]],
            ['<20%', zone_error_rate[3]],
            ['<40%', zone_error_rate[4]],
            ['A', zone_PEG[0]],
            ['B', zone_PEG[1]],
            ['C', zone_PEG[2]],
            ['D', zone_PEG[3]],
            ['E', zone_PEG[4]],
            ['R2',R2],
            ['RMSE', RMSE],
            ['MAE', MAE],
            ['MAPE', MAPE]
        ]
        table = plt.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center', colLoc='center')
        table.scale(1, 2)  # 调整表格的列宽和行高
        return zone_PEG, np.array([R2,RMSE,MAE,MAPE])
            
    def __result_summary(self):
        self.id_row[self.__n_round,0,0] = self.__n_round
        self.id_row[self.__n_round,0,1] = self.__n_round
        self.id_row[self.__n_round,0,2] = self.__n_round

        self.r2_RMSE_MAE_MAPE[self.__n_round,:,0] = self.__r2_RMSE_MAE_MAPE_train
        self.r2_RMSE_MAE_MAPE[self.__n_round,:,1] = self.__r2_RMSE_MAE_MAPE_validation
        self.r2_RMSE_MAE_MAPE[self.__n_round,:,2] = self.__r2_RMSE_MAE_MAPE_test
        
        self.error_percentage[self.__n_round,:,0] = self.__zone_error_rate_train
        self.error_percentage[self.__n_round,:,1] = self.__zone_error_rate_validation
        self.error_percentage[self.__n_round,:,2] = self.__zone_error_rate_test
        
        self.zone_percentage[self.__n_round,:,0] = self.__zone_PEG_train
        self.zone_percentage[self.__n_round,:,1] = self.__zone_PEG_validation
        self.zone_percentage[self.__n_round,:,2] = self.__zone_PEG_test
        
    def __save_result_to_csv(self):
        excel_sheet_name = ['Training Set', 'Validation Set', 'Testing Set']

        # 创建保存路径
        save_path = os.path.join(self.save_path, 'summary.xlsx')

        with pd.ExcelWriter(save_path) as writer:
            for j in range(len(excel_sheet_name)):
                # 最后一轮的 id_row 设置为 '0'
                self.id_row[-2, 0, j] = '0'
                self.id_row[-1, 0, j] = '0'
                
                # 计算 r2, RMSE, MAE, MAPE 的均值和标准差
                for i in range(4):
                    self.r2_RMSE_MAE_MAPE[-2, i, j] = self.r2_RMSE_MAE_MAPE[:self.__n_round, i, j].mean()
                    self.r2_RMSE_MAE_MAPE[-1, i, j] = self.r2_RMSE_MAE_MAPE[:self.__n_round, i, j].std()

                # 计算误差百分比和区域划分的均值和标准差
                for i in range(5):
                    self.error_percentage[-2, i, j] = self.error_percentage[:self.__n_round, i, j].mean()
                    self.error_percentage[-1, i, j] = self.error_percentage[:self.__n_round, i, j].std()
                    self.zone_percentage[-2, i, j] = self.zone_percentage[:self.__n_round, i, j].mean()
                    self.zone_percentage[-1, i, j] = self.zone_percentage[:self.__n_round, i, j].std()
                
                # 将所有结果堆叠成一个矩阵
                info = np.hstack((
                    self.id_row[:, :, j], 
                    self.r2_RMSE_MAE_MAPE[:, :, j], 
                    self.error_percentage[:, :, j], 
                    self.zone_percentage[:, :, j]
                ))
                
                # 创建列名称
                columns = ['No', 'R2', 'RMSE', 'MAE', 'MAPE', 
                        '5% Error', '10% Error', '15% Error', '20% Error', '40% Error', 
                        'Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
                
                # 将矩阵转换为DataFrame
                s = pd.DataFrame(info, columns=columns)
                print(s)
                # 将DataFrame保存到相应的Excel表格
                _replace_dataframe_sheet(writer.book, excel_sheet_name[j], s)

        print(f"Results saved to {save_path}")
