import os
import pandas as pd

def main():
    dirname = os.path.dirname(__file__)
    labelsFile = os.path.join(dirname, '../assets/spcup_2022_training_part1/labels.csv')
    df = pd.read_csv(labelsFile)
    labels_train, labels_dev, labels_test = df['algorithm'].values[:4500], df['algorithm'].values[4500:4750], df['algorithm'].values[4750:]
    files_train, files_dev, files_test = df['track'].values[:4500], df['track'].values[4500:4750], df['track'].values[4750:]
    dict_train = {'track': ['../assets/spcup_2022_training_part1/' + file for file in files_train], 'algorithm': labels_train}
    dict_dev = {'track': ['../assets/spcup_2022_training_part1/' + file for file in files_dev], 'algorithm': labels_dev}
    dict_test = {'track': ['../assets/spcup_2022_training_part1/' + file for file in files_test], 'algorithm': labels_test}
    df_train = pd.DataFrame(dict_train)
    df_dev = pd.DataFrame(dict_dev)
    df_test = pd.DataFrame(dict_test)
    df_train.to_csv(os.path.join(dirname, '../assets/csv/train.csv'), index=False)
    df_dev.to_csv(os.path.join(dirname, '../assets/csv/dev.csv'), index=False)
    df_test.to_csv(os.path.join(dirname, '../assets/csv/test.csv'), index=False)

if __name__=='__main__':
    try: 
        main()
        print("CSV files generated")
    except: print("[Error] Could not generate CSV files")