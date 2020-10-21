"""
Burst extraction pipeline
Kenzie Nimmo 2020
"""

#Setting variables
IN_DIR = "/data1/nimmo/test_AO/"
FILE_BASENAME="$1" #given as input
#the output from the search pipeline should be IN_DIR/FILE_BASENAME/
OUT_DIR="$IN_DIR/$FILE_BASENAME/pulses"
SCRIPT_DIR = "~/AO_burst_properties_pipeline"

cd $OUT_DIR

python $SCRIPT_DIR/RFI_zapper.py $FILE_BASENAME
