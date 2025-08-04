# smiles_variation_generator.py
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

# Disable RDKit logging
import rdkit
rdkit.RDLogger.DisableLog('rdApp.*')

class SmilesVariationGenerator:
    """
    A class to generate non-canonical SMILES variations for molecules.

    This can be used to augment training data for the MST model by providing
    multiple non-canonical representations of the same molecule.
    """

    def __init__(self, seed=42):
        """
        Initialize the SMILES variation generator.

        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def generate_variations(self, smiles, num_variations=20, max_attempts=1000):
        """
        Generate different non-canonical SMILES representations of the same molecule.

        Args:
            smiles (str): Input SMILES string
            num_variations (int): Number of different SMILES strings to generate

        Returns:
            list: List of different SMILES representations
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["Invalid SMILES input"]

        # Get the canonical form for reference
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        variations = set()
        variations.add(canonical_smiles)  # Include the canonical form

        # Method 1: Use random SMILES generation
        for i in range(num_variations * 3):  # Try more times than needed as some might be duplicates
            if len(variations) >= num_variations:
                break
            random.seed(self.seed + i)  # Change seed for each attempt
            random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False, allBondsExplicit=False)
            variations.add(random_smiles)

        # Method 2: Change atom ordering
        for i in range(min(5, num_variations)):
            if len(variations) >= num_variations:
                break
            random.seed(self.seed + i + 100)  # Different seed range
            mol = Chem.MolFromSmiles(smiles)
            atoms = list(range(mol.GetNumAtoms()))
            random.shuffle(atoms)
            random_mol = Chem.RenumberAtoms(mol, atoms)
            variations.add(Chem.MolToSmiles(random_mol, canonical=False, allBondsExplicit=False))

        # Method 3: Change starting atom and bond representation
        for i in range(min(5, num_variations)):
            if len(variations) >= num_variations:
                break
            random.seed(self.seed + i + 200)  # Different seed range
            mol = Chem.MolFromSmiles(smiles)
            start_atom = random.randint(0, mol.GetNumAtoms()-1)
            random_smiles = Chem.MolToSmiles(mol, rootedAtAtom=start_atom, canonical=False, 
                                                allBondsExplicit=bool(i % 2))
            variations.add(random_smiles)

        # Method 4: Generate SMARTS with different features
        smarts_options = [
            (True, True, True, True),    # kekuleSmiles, allBondsExplicit, allHsExplicit, isomericSmiles
            (False, True, False, True),  # Different combination
            (True, False, True, True),   # Different combination
            (False, False, False, True), # Different combination
        ]

        for i, options in enumerate(smarts_options):
            if len(variations) >= num_variations:
                break
            kekule, allBonds, allHs, isomeric = options
            try:
                variant = Chem.MolToSmiles(mol, kekuleSmiles=kekule, allBondsExplicit=allBonds, 
                                            allHsExplicit=allHs, isomericSmiles=isomeric, canonical=False)
                variations.add(variant)
            except:
                continue

        # Convert set to list and ensure we have the requested number of variations
        result = list(variations)

        # If we need more variations, generate some with truly randomized atom ordering
        attempts = 0
        max_remaining_attempts = max_attempts // 4  # Reserve 1/4 of max attempts for this method

        while len(result) < num_variations and attempts < max_remaining_attempts:
            try:
                attempts += 1
                random.seed(len(result) + self.seed + 500 + attempts)  # Different seed
                mol = Chem.MolFromSmiles(smiles)
                atoms = list(range(mol.GetNumAtoms()))
                random.shuffle(atoms)
                random_mol = Chem.RenumberAtoms(mol, atoms)
                new_smiles = Chem.MolToSmiles(random_mol, doRandom=True, canonical=False, 
                                                allBondsExplicit=bool(random.randint(0, 1)))
                if new_smiles not in result:
                    result.append(new_smiles)
            except:
                # If we encounter an error, just continue to the next attempt
                continue

        return result[:num_variations]

    def augment_dataset(self, df, smiles_column='SMILES', num_variations=20, expand=True):
        """
        Augment a dataset by generating non-canonical SMILES variations.

        Args:
            df (pd.DataFrame): Input dataframe containing SMILES
            smiles_column (str): Name of the column containing SMILES strings
            num_variations (int): Number of variations to generate per molecule
            expand (bool): If True, expands the dataset with all variations
                            If False, adds variations as a new column

        Returns:
            pd.DataFrame: Augmented dataframe
        """
        if expand:
            # Create an expanded dataframe with all variations
            expanded_rows = []

            for _, row in df.iterrows():
                smiles = row[smiles_column]
                variations = self.generate_variations(smiles, num_variations)

                # Create a new row for each variation
                for var in variations:
                    new_row = row.copy()
                    new_row[smiles_column] = var
                    expanded_rows.append(new_row)

            return pd.DataFrame(expanded_rows)
        else:
            # Add variations as a new column
            variations_list = []

            for _, row in df.iterrows():
                smiles = row[smiles_column]
                variations = self.generate_variations(smiles, num_variations)
                variations_list.append(variations)

            df_copy = df.copy()
            df_copy['smiles_variations'] = variations_list
            return df_copy

# Example usage
if __name__ == "__main__":
    # Example SMILES - aspirin
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    # Create generator
    generator = SmilesVariationGenerator(seed=42)

    # Generate variations
    variations = generator.generate_variations(test_smiles, num_variations=30)

    # Display results
    print(f"Original SMILES: {test_smiles}")
    print(f"Generated {len(variations)} different SMILES representations:")
    for i, var in enumerate(variations):
        print(f"{i+1}. {var}")

    # Create a sample dataframe
    df = pd.DataFrame({
        'SMILES': [test_smiles, "CCO", "c1ccccc1"],
        'Name': ['Aspirin', 'Ethanol', 'Benzene']
    })

    # Augment the dataframe
    augmented_df = generator.augment_dataset(df, num_variations=5)

    # Display the augmented dataframe
    print("\nAugmented DataFrame:")
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")