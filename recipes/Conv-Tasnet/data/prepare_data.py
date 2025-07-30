"""
The .csv preparation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020
 * Modified by [Your Name] to support dual-channel input

 """

import csv
import os


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath:
        create_wsj_csv(datapath, savepath)
        # if n_spks == 2:
        #     assert (
        #         "2speakers" in datapath
        #     ), "Inconsistent number of speakers and datapath"
        #     create_wsj_csv(datapath, savepath)
        # elif n_spks == 3:
        #     assert (
        #         "3speakers" in datapath
        #     ), "Inconsistent number of speakers and datapath"
        #     create_wsj_csv_3spks(datapath, savepath)
        # else:
        #     raise ValueError("Unsupported Number of Speakers")
    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture1": "mixture1",  # 第一个单通道混合音频文件夹
        "mixture2": "mixture2"   # 第二个单通道混合音频文件夹
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    for set_type in set_types:
        mix1_path = os.path.join(datapath, set_type, folder_names["mixture1"])
        mix2_path = os.path.join(datapath, set_type, folder_names["mixture2"])
        s1_path = os.path.join(datapath, set_type, folder_names["source1"])
        s2_path = os.path.join(datapath, set_type, folder_names["source2"])
        s3_path = os.path.join(datapath, set_type, "source3")  # 假设存在 source3 文件夹
        s4_path = os.path.join(datapath, set_type, "source4")  # 假设存在 source4 文件夹

        files = os.listdir(mix1_path)

        mix1_fl_paths = [os.path.join(mix1_path, fl) for fl in files]
        mix2_fl_paths = [os.path.join(mix2_path, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_path, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_path, fl) for fl in files]
        s3_fl_paths = [os.path.join(s3_path, fl) for fl in files]
        s4_fl_paths = [os.path.join(s4_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav1",  # 第一个单通道混合音频
            "mix_wav1_format",
            "mix_wav1_opts",
            "mix_wav2",  # 第二个单通道混合音频
            "mix_wav2_format",
            "mix_wav2_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
            "s4_wav",
            "s4_wav_format",
            "s4_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"),
            "w",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix1_path, mix2_path, s1_path, s2_path, s3_path, s4_path) in enumerate(
                zip(mix1_fl_paths, mix2_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths, s4_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav1": mix1_path,
                    "mix_wav1_format": "wav",
                    "mix_wav1_opts": None,
                    "mix_wav2": mix2_path,
                    "mix_wav2_format": "wav",
                    "mix_wav2_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "s4_wav": s4_path,
                    "s4_wav_format": "wav",
                    "s4_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix1/")  # 假设存在 mix1 文件夹
        mix2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix2/")  # 假设存在 mix2 文件夹
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")
        s4_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s4/")

        files = os.listdir(mix1_path)

        mix1_fl_paths = [mix1_path + fl for fl in files]
        mix2_fl_paths = [mix2_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]
        s4_fl_paths = [s4_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav1",
            "mix_wav1_format",
            "mix_wav1_opts",
            "mix_wav2",
            "mix_wav2_format",
            "mix_wav2_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
            "s4_wav",
            "s4_wav_format",
            "s4_wav_opts",
        ]

        with open(
            savepath + "/wsj_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix1_path, mix2_path, s1_path, s2_path, s3_path, s4_path) in enumerate(
                zip(mix1_fl_paths, mix2_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths, s4_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav1": mix1_path,
                    "mix_wav1_format": "wav",
                    "mix_wav1_opts": None,
                    "mix_wav2": mix2_path,
                    "mix_wav2_format": "wav",
                    "mix_wav2_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "s4_wav": s4_path,
                    "s4_wav_format": "wav",
                    "s4_wav_opts": None,
                }
                writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-3mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    for set_type in ["tr", "cv", "tt"]:
        mix1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix1/")  # 假设存在 mix1 文件夹
        mix2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/mix2/")  # 假设存在 mix2 文件夹
        s1_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s1/")
        s2_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s2/")
        s3_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s3/")
        s4_path = os.path.join(datapath, "wav8k/min/" + set_type + "/s4/")

        files = os.listdir(mix1_path)

        mix1_fl_paths = [mix1_path + fl for fl in files]
        mix2_fl_paths = [mix2_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        s3_fl_paths = [s3_path + fl for fl in files]
        s4_fl_paths = [s4_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav1",
            "mix_wav1_format",
            "mix_wav1_opts",
            "mix_wav2",
            "mix_wav2_format",
            "mix_wav2_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "s3_wav",
            "s3_wav_format",
            "s3_wav_opts",
            "s4_wav",
            "s4_wav_format",
            "s4_wav_opts",
        ]

        with open(
            savepath + "/wsj_" + set_type + ".csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix1_path, mix2_path, s1_path, s2_path, s3_path, s4_path) in enumerate(
                zip(mix1_fl_paths, mix2_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths, s4_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav1": mix1_path,
                    "mix_wav1_format": "wav",
                    "mix_wav1_opts": None,
                    "mix_wav2": mix2_path,
                    "mix_wav2_format": "wav",
                    "mix_wav2_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "s3_wav": s3_path,
                    "s3_wav_format": "wav",
                    "s3_wav_opts": None,
                    "s4_wav": s4_path,
                    "s4_wav_format": "wav",
                    "s4_wav_opts": None,
                }
                writer.writerow(row)