import os
import argparse
import polars as pl

def get_num_sequences_in_a2m(filepath):
    """Counts the number of sequences in an A2M file. (This function is still useful for initial check)"""
    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # In A2M/FASTA, sequence headers start with '>'
                if line.startswith('>'):
                    count += 1
    except FileNotFoundError:
        print(f"Error: MSA file not found at {filepath}")
        return 0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0
    return count

def read_msa_to_polars(filepath):
    """
    Reads an MSA file (A2M/FASTA) and returns a Polars DataFrame
    with 'id' and 'Sequence' columns, preserving original sequences.
    """
    sequences = []
    current_id = None
    current_seq_parts = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and current_seq_parts:
                        sequences.append((current_id, "".join(current_seq_parts)))
                    
                    # Start new sequence record
                    current_id = line[1:] # Remove the '>'
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line.replace('.', '-').upper())
            
            # Add the last sequence after the loop finishes
            if current_id and current_seq_parts:
                sequences.append((current_id, "".join(current_seq_parts)))

    except FileNotFoundError:
        raise ValueError(f"Error: MSA file not found at {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading {filepath}: {e}")

    # Create Polars DataFrame
    df = pl.DataFrame(sequences, schema={"Entry": pl.String, "Sequence": pl.String})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read MSA and save sequences to parquet.")
    parser.add_argument('--msa-path', type=str, help='Path to the input MSA (.a2m/.afa/.fasta)')
    parser.add_argument('--output-parquet', type=str, help='Path to output .parquet file')
    args = parser.parse_args()

    df = read_msa_to_polars(args.msa_path)
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Write to parquet
    df.write_parquet(args.output_parquet)
    print(f"Saved {len(df)} sequences to {args.output_parquet}")
