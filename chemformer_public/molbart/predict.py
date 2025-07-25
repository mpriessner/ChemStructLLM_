import hydra
import pandas as pd

import molbart.utils.data_utils as util
from molbart.models import Chemformer


def write_predictions(args, smiles, log_lhs, original_smiles):
    num_data = len(smiles)
    #print(smiles)
    smiles_len = [len(i) for i in smiles] ###
    beam_width = max(smiles_len)
    beam_width = len(smiles[0])
    beam_outputs = [[[]] * num_data for _ in range(beam_width)]
    beam_log_lhs = [[[]] * num_data for _ in range(beam_width)]
    #import IPython, sys; IPython.embed();

    for b_idx, (smiles_beams, log_lhs_beams) in enumerate(zip(smiles, log_lhs)):
        for beam_idx, (smi, log_lhs) in enumerate(zip(smiles_beams, log_lhs_beams)):
            beam_outputs[beam_idx][b_idx] = smi
            beam_log_lhs[beam_idx][b_idx] = log_lhs

    df_data = {"target_smiles": original_smiles}
    for beam_idx in range(beam_width):
        df_data["sampled_smiles_" + str(beam_idx + 1)] = beam_outputs[beam_idx]

    for beam_idx in range(beam_width):
        df_data["loglikelihood_" + str(beam_idx + 1)] = beam_log_lhs[beam_idx]

    df = pd.DataFrame(data=df_data)
    df.to_csv(args.output_sampled_smiles, sep="\t", index=False)


@hydra.main(version_base=None, config_path="/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MMT_identifier/chemformer_public/experiment", config_name="project_config")
def main(args):
    chemformer = Chemformer(args)

    print("Making predictions...")
    smiles, log_lhs, original_smiles = chemformer.predict(
        dataset=args.dataset_part,
    )
    write_predictions(args, smiles, log_lhs, original_smiles)
    print("Finished predictions.")
    return


if __name__ == "__main__":
    main()
