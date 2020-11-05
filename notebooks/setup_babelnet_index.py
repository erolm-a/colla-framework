#!/usr/bin/env python

"""
This script sets up the correct index path for BabelNet.
"""

import os
from tools.dumps import get_filename_path

colla_babelnet_key = os.environ["COLLA_BABELNET_KEY"]
colla_babelnet_index = os.environ.get("COLLA_BABELNET_INDEX", get_filename_path("babelnet/BabelNet-4.0.1"))

with open("config/babelnet.var.properties", "w") as fp:
    fp.write(f"babelnet.key={colla_babelnet_key}\n")
    fp.write(f"babelnet.restfulurl=http://babelnet.io/v5/service")
    fp.write(f"babelnet.dir={colla_babelnet_index}")

print("The BabelNet config path has been updated")
