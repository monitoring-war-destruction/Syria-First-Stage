echo "Running experiments for features: ${1}"
for var in "$@"
do
    if [ $var = $1 ]; then
        continue
    fi
    nohup python scripts/validate.py --features=$1 --gpu=$var > validate_${var}.out &
    echo "Started build with gpu ${var}"
    sleep 20
done
