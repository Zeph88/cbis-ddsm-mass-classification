from src.config import MAMMOGRAM_KEY


def validate_columns(dataframe, required_columns, dataframe_name):

    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {dataframe_name}: {missing_columns}")



def build_global_lookup(global_dataframe, extra_columns=None, rename_columns=None):

    extra_columns = ([] if extra_columns is None else list(extra_columns))
    rename_columns = ({} if rename_columns is None else dict(rename_columns))
    required_columns = (MAMMOGRAM_KEY + ["preprocessed_image_path"] + extra_columns)
    validate_columns(global_dataframe, required_columns, "global dataframe")
    global_path_count = (global_dataframe.groupby(MAMMOGRAM_KEY)["preprocessed_image_path"].nunique())
    conflicting_global_paths = global_path_count[global_path_count > 1]

    if not conflicting_global_paths.empty:
        raise ValueError(
            f"Some mammogram keys refer to several different global image paths: {conflicting_global_paths.head()}")

    columns = (MAMMOGRAM_KEY + ["preprocessed_image_path"] + extra_columns)
    rename_mapping = {"preprocessed_image_path": "global_path", **rename_columns}

    return global_dataframe[columns].drop_duplicates(subset=MAMMOGRAM_KEY).rename(columns=rename_mapping)



def pair_local_global(local_dataframe, global_dataframe, how="inner", global_extra_columns=None, global_rename_columns=None, sort=False):
    
    validate_columns(local_dataframe, MAMMOGRAM_KEY + ["preprocessed_image_path"], "local dataframe")
    local_dataframe = local_dataframe.copy()
    local_dataframe["local_path"] = (local_dataframe["preprocessed_image_path"])
    global_lookup = build_global_lookup(global_dataframe=global_dataframe, extra_columns=global_extra_columns, rename_columns=global_rename_columns)
    paired_dataframe = local_dataframe.merge(global_lookup, on=MAMMOGRAM_KEY, how=how, validate="many_to_one", sort=sort)

    return paired_dataframe