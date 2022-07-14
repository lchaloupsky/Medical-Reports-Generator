import shutil
import argparse
import pandas as pd
import json

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default="IU-XRay/images", type=str, help="X-ray images dir.")
parser.add_argument('--models_outputs', default="models_outputs", type=str, help="Folder where outputs are saved.")
parser.add_argument('--reports_orig_dir', default="../translations_openi_cubbit_original/data/ecgen-radiology", type=str, help="Original reports destination.")
parser.add_argument('--reports_trans_dir', default="../translations_openi_cubbit_cased/data/ecgen-radiology", type=str, help="Translated reports destination.")
parser.add_argument('--csv_file', default="testing_set_cz.csv", type=str, help="Path to the testing csv file.")

def prepare(args: argparse.Namespace):
    models_out_dir = Path(args.models_outputs)
    eval_dir = Path("./eval")
    eval_dir.mkdir(exist_ok=True)

    # prepare both sets randomly
    testing_set = pd.read_csv(args.csv_file, encoding="utf-8")
    normal = testing_set.loc[testing_set["Manual Tags"] == "normal"].sample(n=20, replace=False, random_state=0)
    with_disease = testing_set.loc[testing_set["Manual Tags"].str.split(",").str.len() >= 2].sample(n=20, replace=False, random_state=0)

    # structures for holding the data
    norm_dict, norm_map = dict(), dict()
    with_dis_dict, with_dis_map = dict(), dict()

    for mod_dir in models_out_dir.iterdir():
        if not mod_dir.is_dir() or mod_dir.parts[-1] == "normal" or mod_dir.parts[-1] == "with_disease":
            continue
        
        # get current predictions
        preds = pd.read_csv(mod_dir/"predictions.csv", encoding="utf-8")

        # prepare for each model
        prepare_df_set(normal, mod_dir.parts[-1], preds, norm_dict, norm_map)
        prepare_df_set(with_disease, mod_dir.parts[-1], preds, with_dis_dict, with_dis_map)

    # save both normal and with diseases
    save_preds(eval_dir/"normal", norm_dict, norm_map, args)
    save_preds(eval_dir/"with_disease", with_dis_dict, with_dis_map, args)

def save_preds(mod_dir, fin_dict, map_index_img, args):
    for k, report in fin_dict.items():
        final_path_dir = mod_dir/str(k)
        final_path_dir.mkdir(exist_ok=True, parents=True)

        # save all neccessary files
        with open(final_path_dir/"predictions.txt", "wt", encoding="utf-8") as p:
            p.write(report)
        shutil.copyfile(Path(args.img_dir)/map_index_img[k], final_path_dir/f"{str(k)}.png")
        shutil.copyfile(Path(args.reports_orig_dir)/f"{map_index_img[k].split('_')[0].replace('CXR', '')}.txt", final_path_dir/"full_original_english.txt")
        shutil.copyfile(Path(args.reports_trans_dir)/f"{map_index_img[k].split('_')[0].replace('CXR', '')}.txt", final_path_dir/"full_original_czech_translated.txt")

    with open(final_path_dir.parent/"mapping_index_to_img.json", "w") as map_f:
        json.dump(map_index_img, map_f)

def prepare_df_set(df, mod_name, preds, fin_dict, map_index_img):
    for index, (_, row_n) in enumerate(df.iterrows(), 1):
        _, pred_row = next(preds.loc[preds["image_path"] == row_n["Image Index"]].iterrows())
        assert pred_row["image_path"] == row_n["Image Index"]

        # add original if it is not already present
        if index not in fin_dict:
            fin_dict[index] = f"---- Original Translated Czech:\n{pred_row['real']}\n\n"

        # all corresponding preds must have equal base real prediction
        assert pred_row['real'] in fin_dict[index]

        # append current prediction
        fin_dict[index] = fin_dict[index] + f"---- Model: {str(mod_name)}:\n{pred_row['prediction']}\n\n"
        map_index_img[index] = pred_row["image_path"]

if __name__ == "__main__":
    prepare(parser.parse_args([] if "__file__" not in globals() else None))