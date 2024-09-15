import time
import argparse
import pandas as pd
import sentencepiece as spm

original_pretrain_file = 'generated_polymer_smiles_train'

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

# read pretrain file
file_path = 'pretrain_info.csv'
df = pd.read_csv(file_path)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    # Parse the arguments
    args = parser.parse_args()
    size=args.size

    # Record the start time on CPU
    start = time.process_time()

    spm.SentencePieceTrainer.train(input=f'{original_pretrain_file}_{size}.txt',
                                model_prefix=f'spm_{size}',
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
    df.loc[df['pretrain size'] == size, 'tokeniser train time (CPU)'] = cpu_time
    df.to_csv('pretrain_info.csv', index=False)



if __name__ == '__main__':
    main()

