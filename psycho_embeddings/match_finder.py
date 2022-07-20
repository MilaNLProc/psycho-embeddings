from functools import partial
from itertools import repeat
from tqdm import tqdm
from multiprocessing import Pool
import gc
import pandas as pd


if __name__ == "__main__":

    stimuli_file = pd.read_csv("stimuli.csv")
    stimuli = set(stimuli_file["words"].values.tolist())

    with open("corpus.txt") as filino:
        data = filino.readlines()
    data = [k.lower().replace('\n', ' ') for k in data]

    def batch_gen(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


    def func(line):
        collected = []
        for word in stimuli:
            if word in line.split():
                collected.append((word, line))
        return collected


    for index, batch in enumerate(batch_gen(data, 500_000)):

        with Pool(16) as pool:
            matches = list(tqdm(pool.imap(func, batch), total=500_000, position=0))

        with open(f"batches_processed/spp_blp_matches_against_coca_{index}.tsv", "w") as filino:
            for k in matches:
                for key, value in k:
                    filino.write(f"{key}\t{value.strip()}\n")

        del matches
        gc.collect()
