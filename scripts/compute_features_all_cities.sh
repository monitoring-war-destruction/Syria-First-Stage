#!/bin/bash 
if [ "$1" == "--wait-for-aleppo" ]
then
    WAIT_FOR_ALEPPO=true
else
    WAIT_FOR_ALEPPO=false
fi

CITIES=('aleppo' 'idlib' 'damascus' 'homs' 'hama' 'daraa' 'raqqa' 'deir-ez-zor')
for city in ${CITIES[@]}
do
    if [[ ($city == 'aleppo') && ($WAIT_FOR_ALEPPO = true) ]]
    then
        nohup python scripts/compute_features.py --city=$city --filename=${city}.p > "features_${city}.out"
    else
        nohup python scripts/compute_features.py --city=$city --filename=${city}.p > "features_${city}.out" &
    fi
done
