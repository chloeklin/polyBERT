import time
import argparse
import pandas as pd
import os
import platform
import sentencepiece as spm

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

small_elements = [i.lower() for i in elements]

special_tokens =[
    "<pad>",
    "<mask>",
    "[*]",
    "(", ")", "=", "@", "#",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "-", "+",
    "/", "\\",
    "%", "[", "]",
]

special_tokens += elements + small_elements

def train(root_dir: str, size: str):
    # Record the start time on CPU
    start = time.process_time()

    spm.SentencePieceTrainer.train(input=f'{root_dir}/data/pretrain/psmiles_train_{size}.txt',
                                model_prefix=f'{root_dir}/tokeniser/spm/spm_{size}',
                                vocab_size=265,
                                input_sentence_size=5_000_000,
                                #    shuffle_input_sentence=True, # data set is already shuffled
                                character_coverage=1,
                                user_defined_symbols=special_tokens,
                                )
    # Record the end time on CPU
    end = time.process_time()

    # Calculate and print the elapsed time in milliseconds
    cpu_time = end - start

    # Device info (simplified for now)
    device = "CPU (" + platform.processor() + ")"

    # Result to record
    result = {
        'size': size,
        'cpu_time': round(cpu_time, 2),
        'device': device
    }

    # File to store results
    result_file = os.path.join(root_dir, 'tokeniser_training_times.csv')

    # Append to CSV
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(result_file, index=False)
    print(f"Training time recorded to: {result_file}")


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of repository')
    parser.add_argument('--size', type=str, help='Pretraining size')
    # Parse the arguments
    args = parser.parse_args()
    train(args.root_dir, args.size)


if __name__ == '__main__':
    main()

