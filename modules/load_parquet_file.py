import pandas as pd

def load(parquet_file):
    chunks = []
    for batch in parquet_file.iter_batches(batch_size=10000):
        chunks.append(batch.to_pandas())
    df=pd.concat(chunks, ignore_index=True)
    return
