#/bin/bash

# This script was copied from the 3rd sample script. I had assumed that the 
# net included an image recognizer and it would synthesize the image based
# on what it identified. I was wrong. It turned a picture of towers in to a
# bell pepper.

opt_layer=fc6
act_layer=fc8
units=945 # 945 refers to line 945 in the text file. it refers to bell pepper
xy=0

net_weights="nets/placesCNN/places205CNN_iter_300000.caffemodel"
net_definition="nets/placesCNN/places205CNN_deploy_updated.prototxt"

# Hyperparam settings for visualizing AlexNet
iters="30"
weights="99"
rates="1.0"
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file=images/auto_towers.jpg

# Debug
debug=1
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="output"
#rm -rf ${output_dir}
mkdir -p ${output_dir}

# Running optimization across a sweep of hyperparams
for unit in ${units}; do

  for seed in {0..0}; do
  #for seed in {0..8}; do

    for n_iters in ${iters}; do
      for w in ${weights}; do
        for lr in ${rates}; do

          L2="0.${w}"

          # Optimize images maximizing fc8 unit
          python ./act_max.py \
              --act_layer ${act_layer} \
              --opt_layer ${opt_layer} \
              --unit ${unit} \
              --xy ${xy} \
              --n_iters ${n_iters} \
              --start_lr ${lr} \
              --end_lr ${end_lr} \
              --L2 ${L2} \
              --seed ${seed} \
              --clip ${clip} \
              --bound ${bound_file} \
              --debug ${debug} \
              --output_dir ${output_dir} \
              --init_file ${init_file}
        done
      done
    done
  
  done
done

if [ "${debug}" -eq "1" ]; then
  output_file=${output_dir}/towers_placesnet.jpg
  montage debug/*.jpg -tile 10x20 -geometry +1+1 ${output_file}
  convert ${output_file} -trim ${output_file}

  echo "Intermediate results: ${output_file}"
fi
