import gzip
import shutil

input_path = 'PoliMorf-0.6.7.tab.gz'
output_path = 'PoliMorf.tab'

with gzip.open(input_path, 'rb') as f_in:
    with open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Decompressed file saved as {output_path}")
