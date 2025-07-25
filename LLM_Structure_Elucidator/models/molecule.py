"""
Molecule-related functionality for the LLM Structure Elucidator.
"""
import random
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
from config.settings import SAMPLE_SMILES

class MoleculeHandler:
    @staticmethod
    def generate_random_molecule():
        """Generate a random molecule from the sample SMILES strings."""
        try:
            # Select a random SMILES string
            smiles = random.choice(SAMPLE_SMILES)
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            
            return mol
        except Exception as e:
            print(f"Error generating molecule: {str(e)}")
            return None

    @staticmethod
    def validate_smiles(smiles):
        """Validate a SMILES string by attempting to create a molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    @staticmethod
    def calculate_molecular_weight(mol):
        """Calculate the molecular weight of a molecule."""
        try:
            return Descriptors.ExactMolWt(mol)
        except:
            return None

    @staticmethod
    def generate_2d_image(mol, size=(300, 300)):
        """Generate a 2D image of a molecule."""
        try:
            return Draw.MolToImage(mol, size=size)
        except Exception as e:
            print(f"Error generating 2D image: {str(e)}")
            return None
