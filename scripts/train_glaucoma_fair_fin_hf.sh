#!/bin/bash
TASK=cls
MODEL_TYPE='efficientnet'  # quant, resnext, efficientnet
LOSS_TYPE='bce' 
LR=5e-5
NUM_EPOCH=4
BATCH_SIZE=6
MODALITY_TYPE='rnflt'
ATTRIBUTE_TYPE='race'
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
NORMALIZATION_TYPE=fin
NORMALISE_DATA=0
PERF_FILE=${MODEL_TYPE}_${NORMALIZATION_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python train_glaucoma_fair_fin_hf.py \
	--data_dir ../quant_notes/data_cmpr/ \
	--result_dir ./results/crosssectional_${MODALITY_TYPE}_${NORMALIZATION_TYPE}_${ATTRIBUTE_TYPE}_ablation_of_sigma/fullysup_${MODEL_TYPE}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE}_normdata${NORMALISE_DATA} \
	--model_type ${MODEL_TYPE} \
	--image_size 200 \
	--loss_type ${LOSS_TYPE} \
	--lr ${LR} --weight-decay 0. --momentum 0.1 \
	--batch-size ${BATCH_SIZE} \
	--task ${TASK} \
	--epochs ${NUM_EPOCH} \
	--modality_types ${MODALITY_TYPE} \
	--perf_file ${PERF_FILE} \
	--normalization_type ${NORMALIZATION_TYPE} \
	--attribute_type ${ATTRIBUTE_TYPE} \
	--normalise_data ${NORMALISE_DATA}