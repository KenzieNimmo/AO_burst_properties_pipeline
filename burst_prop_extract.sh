#!/bin/bash -x                                                                                                                                                         

if [ $# -ne 1 ] && [ $# -ne 2 ]; then
   echo "Pipeline to process Arecibo data of FRB121102 on DRAGNET."
   echo "The pipeline extracts the burst properties of FRB 121102 bursts found in pulses_extract pipeline"
   echo ""
   echo "Usage: bash burst_prop_extract.sh fits_filename"
   echo "NB: use the command bash to run the pipeline"
   exit
fi

#Check that bash is used                                                                                                                                               
if [ ! "$BASH_VERSION" ] ; then
    echo "Execute the script using the bash command. Exiting..."
    exit 1
fi

echo "Pipeline burst_prop_extract.sh starting..."
date

#Setting variables
IN_DIR="/data1/nimmo/test_AO/"
FILE_BASENAME="$1" #given as input
#the output from the search pipeline should be IN_DIR/FILE_BASENAME/
OUT_DIR="$IN_DIR/$FILE_BASENAME/pulses"
SCRIPT_DIR="/home/nimmo/AO_burst_properties_pipeline"

cd $OUT_DIR

#python $SCRIPT_DIR/RFI_zapper.py $FILE_BASENAME

echo "2D Gaussian fit"
python $SCRIPT_DIR/fit_bursts.py $FILE_BASENAME

echo "Barycentring TOA"
python $SCRIPT_DIR/TOA_bary.py $FILE_BASENAME

echo "Fluence/Flux/Energy calculations"
python $SCRIPT_DIR/fluence_flux_energy.py -d 972 $FILE_BASENAME #assuming 972Mpc distance

