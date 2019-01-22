from rdkit import Chem

import sys
import os


def molecule_supplier_from_name(input_file_name):
    ext = os.path.splitext(input_file_name)[-1]
    if ext == ".smi":
        suppl = Chem.SmilesMolSupplier(input_file_name, titleLine=False)
    elif ext == ".sdf":
        suppl = Chem.SDMolSupplier(input_file_name)
    else:
        print("%s is not a valid molecule extension" % ext)
        sys.exit(0)
    return suppl

