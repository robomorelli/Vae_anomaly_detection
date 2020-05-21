import pandas as pd
import numpy as np
import uproot
import os
import random
from config import *
from utils import *


proportions = [0.30, 0.30, 0.40]

def main():

    try:
        os.makedirs(train_val_test)
    except:
        pass

    train_val_test_split(numpy_bkg, train_val_test, proportions)

if __name__ == "__main__":

    main()
