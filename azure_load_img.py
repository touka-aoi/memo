from azureml.core import Run

import pandas as pd 
import numpy as np 
import argparse

RANDOM_SEED=42

# コマンドライン引数を取得
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', dest='output_path', required=True)
args = parser.parse_args()  

def load_img(stInfo):
    filename = str(stInfo).split("/")[-1]
    with stInfo.open() as f:
        img = Image.open(f)
        img_numpy = np.array(img)
        img_gray = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
        output_path = os.path.join(args.output_path, filename)
        cv2.imwrite(output_path, img_gray)
        return output_path


ds = Run.get_context().input_datasets['train_ds']
whole_df = ds.to_pandas_dataframe()
url_df = whole_df.iloc[:,[0]]
url_df = url_df.applymap(load_img)
whole_df.iloc[;, 0] = url_df

whole_df.to_csv(os.path.join(args.output_path,"prepped_data.csv"))

print(f"Wrote prepped data to {args.output_path}/prepped_data.csv")

