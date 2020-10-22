#!/usr/bin/env bash

output_dir="${1:-./baseline}"

split_seed="1549786796" # original 1549786796
sampling_seed="1549786595" # original 1549786595


###################### Functions ###################################

function move_data() {
	path="$1"
	suffix="$2"

	pushd models/metrics
		echo "${path}/sys_metrics_${suffix}.csv", "${path}/stat_metrics_${suffix}.csv"
		mv sys_metrics.csv "${path}/sys_metrics_${suffix}.csv"
		mv stat_metrics.csv "${path}/stat_metrics_${suffix}.csv"
	popd

	cp -r data/femnist/meta "${path}"
	mv "${path}/meta" "${path}/meta_${suffix}"
}


##################### Script #################################
# pushd ../

# Check that data and models are available
if [ ! -d 'data/' -o ! -d 'models/' ]; then
	echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# If data unavailable, execute pre-processing script
if [ ! -d 'data/femnist/data/train' ]; then
	if [ ! -f 'data/femnist/preprocess.sh' ]; then
		echo "Couldn't find data/ and/or models/ directories - please obtain scripts from GitHub repo: https://github.com/TalwalkarLab/leaf"
		exit 1
	fi

	echo "Couldn't find FEMNIST data - running data preprocessing script"
	pushd data/femnist/
		rm -rf meta/ data/test data/train data/rem_user_data data/intermediate
		./preprocess.sh -s niid --sf 0.05 -k 100 -t sample --smplseed ${sampling_seed} --spltseed ${split_seed}
	popd
fi

