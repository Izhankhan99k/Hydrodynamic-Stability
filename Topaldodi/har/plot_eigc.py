import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved data
data = np.load("stability_results.npz")

cr_clean = data['cr_clean']
ci_clean = data['ci_clean']
unstable_mask = data['unstable_mask']
stable_mask = data['stable_mask']

# Load parameters for the title
Re = data['Re']
k = data['k']
a = data['a']
G = data['G']
Bo = data['Bo']

# Initialize the plot
