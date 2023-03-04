import os
import re
import pandas as pd
import re
import random


def get_file_list(data_path):
    data_list = []
    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".jpeg")):
                    data_list.append(os.path.join(subdir, file))
    data_list.sort()
    if not data_list:
        raise FileNotFoundError("No data was found")
    return data_list


def train_val_pathes(all_data_pathes, regex_for_category):
    all_pathes = get_file_list(all_data_pathes)
    all_pathes = [data.replace("\\", "/") for data in all_pathes]
    all_pathes = [path for path in all_pathes if re.findall("Output", path) == []]

    data_pathes = pd.DataFrame(all_pathes, columns={"path"})
    data_pathes["category"] = data_pathes["path"].apply(
        lambda x: re.search(
            regex_for_category,
            x,
        ).group(1)
    )
    random_categories = random.sample(list(data_pathes["category"].unique()), k=20)
    #########################     val Dataset     ###############################
    val_path_list = []
    classes = data_pathes.category.unique()
    sample_size = 50
    for category in random_categories:
        g = data_pathes[data_pathes.category == category].sample(sample_size)
        val_path_list.append(g)
    val_pathes = pd.concat(val_path_list)
    #######################     Train Dataset      ##############################
    train_path_index = [
        path for path in list(data_pathes.index) if path not in list(val_pathes.index)
    ]
    train_pathes = data_pathes.loc[train_path_index, :].reset_index(drop=True)
    train_pathes = train_pathes[train_pathes.category.isin(random_categories)]

    return train_pathes, val_pathes, random_categories
