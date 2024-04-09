from pathlib import Path
import pickle
import pandas as pd


def main():
    intent_file_path = Path.cwd().parent / "data" / "RPAB.txt"
    with open(intent_file_path, "r") as f:
        intent_list = f.readlines()
    intent_list = [intent.strip() for intent in intent_list]
    assert len(intent_list) == 50

    # get all filenames in outputs/Llama-2-7b-chat-hf
    output_dir = Path.cwd() / "outputs" / "Llama-2-7b-chat-hf"
    filenames = [file.name for file in output_dir.iterdir()]

    for intent_index in range(50):
        data = {"intent": intent_list[intent_index], "attempts": []}
        # read all files in the output_dir, and get the data
        for filename in filenames:
            assert filename.endswith(".csv")
            with open(output_dir / filename, "r") as f:
                df = pd.read_csv(f)
                # assert dataframe only has 50 rows

            # get the intent_index-th row
            row = df.iloc[intent_index]
            # get the "prompt" column
            intent = row["prompt"]
            assert intent == intent_list[intent_index]

            # get the "output" column
            response = row["output"]

            # extract setting name from filename
            setting = filename[7:-4]

            attempt = {"setting": setting, "response": response, "label": None}

            # append the data to the data["attempts"]
            data["attempts"].append(attempt)

        # create a pickle file for data in output folder
        with open(f"outputs_pickle/{intent_index}.pkl", "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
