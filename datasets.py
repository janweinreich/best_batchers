import random
import numpy as np
import pandas as pd
import torch
from BO_utils import update_model
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import AllChem

# Function to pad the arrays
def pad_array(arr, target_length):
    padding = target_length - len(arr)
    return np.pad(arr, (0, padding), "constant")


def convert2pytorch(X, y, type_X="float"):
    if type_X == "float":
        X = torch.from_numpy(X).float()
    elif type_X == "int":
        X = torch.from_numpy(X).int()
    else:
        raise ValueError("Invalid type for X.")
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


class FingerprintGenerator:
    def __init__(self, nBits=512, radius=2):
        self.nBits = nBits
        self.radius = radius

    def featurize(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                fingerprints.append(np.ones(self.nBits))
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.nBits
                    )
                    fp_array = np.array(
                        list(fp.ToBitString()), dtype=int
                    )  # Convert to NumPy array
                    fingerprints.append(fp_array)
                else:
                    print(f"Could not generate a molecule from SMILES: {smiles}")
                    fingerprints.append(np.array([None]))

        return np.array(fingerprints)


def check_entries_scaling(array_of_arrays):
    """
    Check if the entries of the arrays are between 0 and 1.
    Needed for for the datasets.py script.
    """

    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True


def inchi_to_smiles(inchi_list):
    """
    Convert a list of InChI strings to a list of canonical SMILES strings.

    Args:
    inchi_list (list): A list of InChI strings.

    Returns:
    list: A list of canonical SMILES strings.
    """
    smiles_list = []
    for inchi in inchi_list:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)  # Append None for invalid InChI strings
    return smiles_list


class directaryl:
    def __init__(self):
        # direct arylation reaction
        self.ECFP_size = 512
        self.radius = 4
        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)
        dataset_url = "https://raw.githubusercontent.com/doyle-lab-ucla/edboplus/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full.csv"
        self.data = pd.read_csv(dataset_url)
        self.data = self.data.sample(frac=1, random_state=666).reset_index(drop=True)
        # create a copy of the data
        data_copy = self.data.copy()
        # remove the Yield column from the copy
        data_copy.drop("Yield", axis=1, inplace=True)
        # check for duplicates
        duplicates = data_copy.duplicated().any()
        if duplicates:
            print("There are duplicates in the dataset.")
            exit()

        self.data["Base_SMILES"] = inchi_to_smiles(self.data["Base_inchi"].values)
        self.data["Ligand_SMILES"] = inchi_to_smiles(self.data["Ligand_inchi"].values)
        self.data["Solvent_SMILES"] = inchi_to_smiles(self.data["Solvent_inchi"].values)
        col_0_base = self.ftzr.featurize(self.data["Base_SMILES"])
        col_1_ligand = self.ftzr.featurize(self.data["Ligand_SMILES"])
        col_2_solvent = self.ftzr.featurize(self.data["Solvent_SMILES"])
        col_3_concentration = self.data["Concentration"].to_numpy().reshape(-1, 1)
        col_4_temperature = self.data["Temp_C"].to_numpy().reshape(-1, 1)
        self.X = np.concatenate(
            [
                col_0_base,
                col_1_ligand,
                col_2_solvent,
                col_3_concentration,
                col_4_temperature,
            ],
            axis=1,
        )
        self.experiments = np.concatenate(
            [
                self.data["Base_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Ligand_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Solvent_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Concentration"].to_numpy().reshape(-1, 1),
                self.data["Temp_C"].to_numpy().reshape(-1, 1),
                self.data["Yield"].to_numpy().reshape(-1, 1),
            ],
            axis=1,
        )

        self.y = self.data["Yield"].to_numpy()
        self.all_ligands = self.data["Ligand_SMILES"].to_numpy()
        self.all_bases = self.data["Base_SMILES"].to_numpy()
        self.all_solvents = self.data["Solvent_SMILES"].to_numpy()
        unique_bases = np.unique(self.data["Base_SMILES"])
        unique_ligands = np.unique(self.data["Ligand_SMILES"])
        unique_solvents = np.unique(self.data["Solvent_SMILES"])
        unique_concentrations = np.unique(self.data["Concentration"])
        unique_temperatures = np.unique(self.data["Temp_C"])

        max_yield_per_ligand = np.array(
            [
                max(self.data[self.data["Ligand_SMILES"] == unique_ligand]["Yield"])
                for unique_ligand in unique_ligands
            ]
        )

        self.worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]
        self.best_ligand = unique_ligands[np.argmax(max_yield_per_ligand)]

        self.where_worst_ligand = np.array(
            self.data.index[self.data["Ligand_SMILES"] == self.worst_ligand].tolist()
        )

        self.feauture_labels = {
            "names": {
                "bases": unique_bases,
                "ligands": unique_ligands,
                "solvents": unique_solvents,
                "concentrations": unique_concentrations,
                "temperatures": unique_temperatures,
            },
            "ordered_smiles": {
                "bases": self.data["Base_SMILES"],
                "ligands": self.data["Ligand_SMILES"],
                "solvents": self.data["Solvent_SMILES"],
                "concentrations": self.data["Concentration"],
                "temperatures": self.data["Temp_C"],
            },
        }


class Evaluation_data_directaryl:
    def __init__(self):
        self.get_raw_dataset()

        rep_size = self.X.shape[1]
        self.bounds_norm = torch.tensor([[0] * rep_size, [1] * rep_size])
        self.bounds_norm = self.bounds_norm.to(dtype=torch.float32)

        if not check_entries_scaling(self.X):

            self.scaler_X = MinMaxScaler()
            self.X = self.scaler_X.fit_transform(self.X)

    def get_raw_dataset(self):
        # https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full_update.csv

        BMS = directaryl()
        self.data = BMS.data
        self.experiments = BMS.experiments
        self.X, self.y = BMS.X, BMS.y

        self.all_ligands = BMS.all_ligands
        self.all_bases = BMS.all_bases
        self.all_solvents = BMS.all_solvents

        self.best_ligand = BMS.best_ligand
        self.worst_ligand = BMS.worst_ligand
        self.where_worst_ligand = BMS.where_worst_ligand
        self.feauture_labels = BMS.feauture_labels

    def get_init_holdout_data(self, SEED):
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        indices_init = np.random.choice(
            self.where_worst_ligand[:200], size=48, replace=False
        )
        exp_init = self.experiments[indices_init]
        indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

        np.random.shuffle(indices_init)
        np.random.shuffle(indices_holdout)

        X_init, y_init = self.X[indices_init], self.y[indices_init]
        X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
        exp_holdout = self.experiments[indices_holdout]

        LIGANDS_INIT = self.all_ligands[indices_init]
        LIGANDS_HOLDOUT = self.all_ligands[indices_holdout]

        X_init, y_init = convert2pytorch(X_init, y_init)
        X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

        return (
            X_init,
            y_init,
            X_holdout,
            y_holdout,
            LIGANDS_INIT,
            LIGANDS_HOLDOUT,
            exp_init,
            exp_holdout,
        )


def init_directaryl(seed):
    # Initialize data from dataset
    DATASET = Evaluation_data_directaryl()
    bounds_norm = DATASET.bounds_norm

    (
        X_init,
        y_init,
        X_pool_fixed,
        y_pool_fixed,
        _,
        _,
        _,
        _,
    ) = DATASET.get_init_holdout_data(seed)

    # Construct initial shitty model
    model, _ = update_model(
        X_init,
        y_init,
        bounds_norm,
        kernel_type="Tanimoto",
        fit_y=False,
        FIT_METHOD=True,
    )

    # Copy things to avoid problems later
    X_train = np.copy(X_init)
    y_train = np.copy(y_init)
    X_pool = np.copy(X_pool_fixed)
    y_pool = np.copy(y_pool_fixed)

    return model, X_train, y_train, X_pool, y_pool, bounds_norm


class formed:
    def __init__(self, new_parse=True, SMILES_MODE=True):
        # https://archive.materialscloud.org/record/2022.162
        self.max_N = 2000000
        self.SMILES_MODE = SMILES_MODE
        self.new_parse = new_parse
        self.bounds_norm = None

        if self.new_parse:
            self.data = pd.read_csv("Data_FORMED_scored.csv",
                usecols=["name", "Canonical_Smiles", "gap"],
                delimiter=",",
            )
            self.data.dropna(axis=0, inplace=True)  # Drop rows with NA values
            self.data.reset_index(
                drop=True, inplace=True
            )  # Reset the index and drop the old index
            self.data.drop_duplicates(
                subset="Canonical_Smiles", keep="first", inplace=True, ignore_index=True
            )
            # @Jan check the two above lines, if you'd like keep the initial indices remove 'ignore_index=True'

            self.names = self.data["name"].values
            self.smiles = self.data["Canonical_Smiles"].values
            self.y = self.data["gap"].values

            indices = np.arange(len(self.names))
            np.random.shuffle(indices)
            self.names = self.names[indices]
            self.smiles = self.smiles[indices]
            self.y = self.y[indices]

            self.names = self.names[: self.max_N]
            self.smiles = self.smiles[: self.max_N]
            self.y = self.y[: self.max_N]

            if self.SMILES_MODE:
                self.ECFP_size = 512  # 1024
                self.radius = 4
                self.ftzr = FingerprintGenerator(
                    nBits=self.ECFP_size, radius=self.radius
                )
                self.X = self.ftzr.featurize(self.smiles)
                self.scaler_X = MinMaxScaler()
                self.X = self.scaler_X.fit_transform(self.X)
                
                np.savez_compressed("formed_SMILES.npz",
                    names=self.names,
                    X=self.X,
                    y=self.y,
                    smiles=self.smiles,
                )
            else:
                self.X = []
                keep_inds = []
                for i, name in tqdm(enumerate(self.names)):
                    mol = compound.xyz_to_mol("XYZ_FORMED/{}.xyz".format(name), "def2svp"
                    )
                    spahm_rep = spahm.compute_spahm.get_spahm_representation(mol, "lb")[
                        0
                    ]
                    self.X.append(spahm_rep)
                    keep_inds.append(i)
                max_length = max(len(item) for item in self.X)
                self.X = np.array([pad_array(item, max_length) for item in self.X])

                self.scaler_X = MinMaxScaler()
                self.X = self.scaler_X.fit_transform(self.X)
                np.savez_compressed("formed_SPAHM.npz",
                    names=self.names,
                    X=self.X,
                    y=self.y,
                    smiles=self.smiles,
                )

        else:
            if self.SMILES_MODE:
                data = np.load(
                    "formed_SMILES.npz", allow_pickle=True
                )
            else:
                data = np.load(
                    "formed_SPAHM.npz", allow_pickle=True
                )

            self.names = data["names"]
            self.X = data["X"]
            self.y = data["y"]
            self.smiles = data["smiles"]

    def get_init_holdout_data(self, SEED):
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)


        indices_init = np.random.choice(np.arange(len(self.X)), size=50, replace=False)
        indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

        np.random.shuffle(indices_init)
        np.random.shuffle(indices_holdout)

        X_init, y_init, smiles_init = (
            self.X[indices_init],
            self.y[indices_init],
            self.smiles[indices_init],
        )
        X_holdout, y_holdout, smiles_holdout = (
            self.X[indices_holdout],
            self.y[indices_holdout],
            self.smiles[indices_holdout],
        )

        if max(y_init) > max(y_holdout):
            ind_max = np.argmax(y_init)
            X_holdout = np.vstack([X_holdout, X_init[ind_max]])
            y_holdout = np.append(y_holdout, y_init[ind_max])
            X_init = np.delete(X_init, ind_max, axis=0)
            y_init = np.delete(y_init, ind_max, axis=0)
            smiles_init = np.delete(smiles_init, ind_max, axis=0)
            smiles_holdout = np.append(smiles_holdout, smiles_init[ind_max])

        if self.SMILES_MODE:
            X_init, y_init = convert2pytorch(X_init, y_init, type_X="int")
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout, type_X="int")

        else:
            X_init, y_init = convert2pytorch(X_init, y_init, type_X="float")
            X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout, type_X="float")
        return (X_init, y_init, X_holdout, y_holdout, smiles_init, smiles_holdout)


def init_formed(seed):
    # Initialize data from dataset
    DATASET = formed(new_parse=True, SMILES_MODE=True)
    bounds_norm = DATASET.bounds_norm

    (
        X_init,
        y_init,
        X_pool_fixed,
        y_pool_fixed,
        smiles_init,
        smiles_pool,
    ) = DATASET.get_init_holdout_data(seed)

    # Construct initial shitty model
    model, _ = update_model(
        X_init,
        y_init,
        bounds_norm,
        kernel_type="Tanimoto",
        fit_y=False,
        FIT_METHOD=True,
    )

    # Copy things to avoid problems later
    X_train = np.copy(X_init)
    y_train = np.copy(y_init)
    X_pool = np.copy(X_pool_fixed)
    y_pool = np.copy(y_pool_fixed)

    return model, X_train, y_train, X_pool, y_pool, bounds_norm
