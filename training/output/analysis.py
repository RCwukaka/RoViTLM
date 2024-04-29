import os
from os.path import join

import pandas as pd


def getResult():
    current_file = __file__
    path = join(os.path.abspath(current_file), '../')
    info = {}
    dirs = os.listdir(path)
    dirs.remove('__pycache__')
    dirs.remove('analysis.py')
    for model in dirs:
        model_out_path = os.path.join(path, model)
        w = {"W0_W1": 0, "W0_W2": 0, "W0_W3": 0, "W1_W0": 0, "W1_W2": 0, "W1_W3": 0, "W2_W0": 0, "W2_W1": 0, "W2_W3": 0, "W3_W0": 0, "W3_W1": 0, "W3_W2": 0}
        w_task_path = os.path.join(model_out_path, "W")
        for file in os.listdir(w_task_path):
            if check_file_extension(file, 'xlsx'):
                task_name = file.split(".")[0]
                accuracy = analysis(os.path.join(w_task_path, file))
                w.update({task_name: accuracy})

        t = {"T0_T1": 0, "T0_T2": 0, "T1_T0": 0, "T1_T2": 0, "T2_T0": 0, "T2_T1": 0}
        t_task_path = os.path.join(model_out_path, "T")
        for file in os.listdir(t_task_path):
            if check_file_extension(file, 'xlsx'):
                task_name = file.split(".")[0]
                accuracy = analysis(os.path.join(t_task_path, file))
                t.update({task_name: accuracy})

        p = {"P0_P1": 0, "P0_P2": 0, "P1_P0": 0, "P1_P2": 0, "P2_P0": 0, "P2_P1": 0}
        p_task_path = os.path.join(model_out_path, "P")
        for file in os.listdir(p_task_path):
            if check_file_extension(file, 'xlsx'):
                task_name = file.split(".")[0]
                accuracy = analysis(os.path.join(p_task_path, file))
                p.update({task_name: accuracy})

        h = {"H0_H1": 0, "H0_H2": 0, "H0_H3": 0, "H1_H0": 0, "H1_H2": 0, "H1_H3": 0, "H2_H0": 0, "H2_H1": 0, "H2_H3": 0, "H3_H0": 0, "H3_H1": 0, "H3_H2": 0}
        h_task_path = os.path.join(model_out_path, "H")
        for file in os.listdir(h_task_path):
            if check_file_extension(file, 'xlsx'):
                task_name = file.split(".")[0]
                accuracy = analysis(os.path.join(h_task_path, file))
                h.update({task_name: accuracy})
        info.update({model: {"p": p, "t": t, "w": w, "h": h}})

    w = {}
    p = {}
    t = {}
    h = {}

    w_keys = []
    p_keys = []
    t_keys = []
    h_keys = []
    for model_key in info:
        for task_key in info[model_key]:
            w_accuracy = []
            t_accuracy = []
            p_accuracy = []
            h_accuracy = []
            if task_key == 'w':
                for w_key in info[model_key][task_key]:
                    if w_keys.count(w_key) == 0:
                        w_keys.append(w_key)
                    w_accuracy.append(info[model_key][task_key][w_key])
                w.update({model_key: w_accuracy})

            if task_key == 'p':
                for p_key in info[model_key][task_key]:
                    if p_keys.count(p_key) == 0:
                        p_keys.append(p_key)
                    p_accuracy.append(info[model_key][task_key][p_key])
                p.update({model_key: p_accuracy})

            if task_key == 't':
                for t_key in info[model_key][task_key]:
                    if t_keys.count(t_key) == 0:
                        t_keys.append(t_key)
                    t_accuracy.append(info[model_key][task_key][t_key])
                t.update({model_key: t_accuracy})

            if task_key == 'h':
                for h_key in info[model_key][task_key]:
                    if h_keys.count(h_key) == 0:
                        h_keys.append(h_key)
                    h_accuracy.append(info[model_key][task_key][h_key])
                h.update({model_key: h_accuracy})

    w.update({"TASK": w_keys})
    p.update({"TASK": p_keys})
    t.update({"TASK": t_keys})
    h.update({"TASK": h_keys})
    w_df = pd.DataFrame(w)
    w_df.to_excel(join(os.path.abspath(path), '../W.xlsx'), index=False)
    p_df = pd.DataFrame(p)
    p_df.to_excel(join(os.path.abspath(path), '../P.xlsx'), index=False)
    h_df = pd.DataFrame(h)
    h_df.to_excel(join(os.path.abspath(path), '../H.xlsx'), index=False)
    t_df = pd.DataFrame(t)
    t_df.to_excel(join(os.path.abspath(path), '../T.xlsx'), index=False)

    return


def check_file_extension(filename, extension):
    file_extension = filename.split('.')[-1]
    if file_extension == extension:
        return True
    else:
        return False


def split_file_name(filename):
    task_name = filename.split('\\')[-1].split('.')[0]
    model_name = filename.split('../')[-1].split('\\')[0]
    return task_name, model_name


def analysis(filename):
    if os.path.exists(filename):
        data = pd.read_excel(filename)
        accuracy = (data.at[len(data) - 1, 'accuracy'] + data.at[len(data) - 2, 'accuracy'] + \
                    data.at[len(data) - 3, 'accuracy'] + data.at[len(data) - 4, 'accuracy'] + data.at[
                        len(data) - 5, 'accuracy']) / 5
        return accuracy
    else:
        return 0


if __name__ == '__main__':
    getResult()
