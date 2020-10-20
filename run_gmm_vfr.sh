#!/bin/bash

# SPAPL
# Top script for CSRC in SLT2021

wav_dir=/media/kingformatty/king/SLT2021_CSRC_aug/SLT2021_CSRC_DATA/

datadir=data
mfccdir=mfcc
#vfrdir=/media/kingformatty/king/VFR_feat/SetC_train/split/
expdir=exp_vfraug
stage=12
stop_stage=12

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # Prepare dictionary from aishell1 lexicon
  local/aishell_prepare_dict.sh conf/
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # format the data as Kaldi data directories
  local/data_prepare.sh --part "SetA" $wav_dir $datadir/local
  local/data_prepare.sh --part "SetC1" $wav_dir $datadir/local
  local/data_prepare.sh --part "SetC2" $wav_dir $datadir/local

  local/format_data.sh --dev_ratio 5 $datadir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] ; then
  # Phone sets, L.fst
  utils/prepare_lang.sh --position-dependent-phones false $datadir/local/dict \
    "<SPOKEN_NOISE>" $datadir/local/lang $datadir/lang || exit 1;
  
  # train 3gram language model
  local/aishell_train_lms.sh $datadir/SetA_train/text
  
  # LG composition
  utils/format_lm.sh $datadir/lang $datadir/local/lm/3gram-mincount/lm_unpruned.gz \
    $datadir/local/dict/lexicon.txt $datadir/lang_test || exit 1;
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  for part in SetC1_dev SetC2_dev; do #SetA_dev SetC1_dev SetC2_dev; do
    #steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj 8 \
    #  $datadir/$part exp_aug/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh $datadir/$part $expdir/make_mfcc/$part $mfccdir
    utils/fix_data_dir.sh $datadir/$part
  done
fi



#train=data/SetC1_train
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 180k
  # utterances in the train directory.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.
for train in SetC_train_VFRaug; do  
  utils/subset_data_dir.sh $datadir/$train 30000 $datadir/${train}_30k
  utils/subset_data_dir.sh $datadir/$train 90000 $datadir/${train}_90k
done
fi

#test_set="SetA_dev" #SetC1_dev SetC2_dev"
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then

for train in SetC_train_VFRaug; do
  # train a monophone system
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
    $datadir/${train}_30k $datadir/lang $expdir/mono_${train}_vfr

  # decode using the monophone model
  utils/mkgraph.sh $datadir/lang_test $expdir/mono_${train}_vfr $expdir/mono_${train}/graph
  

  #for test in $test_set; do
  #  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  #    exp/mono/graph data/$test exp/mono/decode_$test
  #done
done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then

for train in SetC_train_VFRaug; do
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    $datadir/${train}_90k $datadir/lang $expdir/mono_${train} $expdir/mono_ali_90k_${train}

  # train a first delta + delta-delta triphone system on a subset of 90000 utterances
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 $datadir/${train}_90k $datadir/lang $expdir/mono_ali_90k_${train} $expdir/tri1_${train}

  # decode using the tri1 model
  utils/mkgraph.sh $datadir/lang_test $expdir/tri1_${train} $expdir/tri1_${train}/graph

  #for test in $test_set; do
  #  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  #    exp/tri1/graph data/$test exp/tri1/decode_$test
  #done
done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
for train in SetC_train_VFRaug; do
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    $datadir/${train} $datadir/lang $expdir/tri1_${train} $expdir/tri1_ali_${train}

  # train a second delta system with all training data
  steps/train_deltas.sh --cmd "$train_cmd" \
    4000 32000 $datadir/${train} $datadir/lang $expdir/tri1_ali_${train} $expdir/tri2b_${train}

  # decode using the tri2b model
  utils/mkgraph.sh $datadir/lang_test $expdir/tri2b_${train} $expdir/tri2b_${train}/graph

  #for test in $test_set; do
  #  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  #    exp/tri2b/graph data/$test exp/tri2b/decode_$test
  #done
done  
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
for train in SetC_train_VFRaug; do 
  # Align
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    $datadir/${train} $datadir/lang $expdir/tri2b_${train} $expdir/tri2b_ali_${train}

  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
    6200 80000 $datadir/${train} $datadir/lang $expdir/tri2b_ali_${train} $expdir/tri3b_${train}

  # decode using the LDA+MLLT model  
  utils/mkgraph.sh $datadir/lang_test $expdir/tri3b_${train} $expdir/tri3b_${train}/graph
  #for test in $test_set; do
  #  steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  #    exp/tri3b/graph data/$test exp/tri3b/decode_$test
  #done
done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
for train in SetC_train_VFRaug; do
  # Align
  steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    $datadir/${train} $datadir/lang_test $expdir/tri3b_${train} $expdir/tri3b_ali_${train}

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" \
    6200 80000 $datadir/${train} $datadir/lang $expdir/tri3b_ali_${train} $expdir/tri4b_${train}

  # decode using the tri3b model
  utils/mkgraph.sh $datadir/lang_test $expdir/tri4b_${train} $expdir/tri4b_${train}/graph
done

fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
for train in SetC_train_VFRaug; do
  for test in SetC1_dev SetC2_dev; do
    steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" \
      $expdir/tri4b_${train}/graph $datadir/$test $expdir/tri4b_${train}/decode_raw_$test
  
done
done

fi


