#!/bin/bash

# SPAPL
# Top script for CSRC in SLT2021

wav_dir=/media/kingformatty/king/SLT2021_CSRC/SLT2021_CSRC_DATA/
#wav_dir=/home/ruchao/Database/SLT2021_CSRC/SLT2021_CSRC_DATA/

datadir=data
mfccdir=mfcc
stage=7
stop_stage=10

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
  #local/data_prepare.sh --part "SetA" $wav_dir $datadir/local
  #local/data_prepare.sh --part "SetC1" $wav_dir $datadir/local
  local/data_prepare.sh --part "SetC2" $wav_dir $datadir/local

  local/format_data.sh --dev_ratio 5 $datadir
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ] && false; then
  # Phone sets, L.fst
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;
  
  # train 3gram language model
  local/aishell_train_lms.sh $datadir/SetA_train/text
  
  # LG composition
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  for part in SetC_train SetC_dev; do #SetA_dev SetC1_dev SetC2_dev; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj 8 \
      $datadir/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh $datadir/$part exp/make_mfcc/$part $mfccdir
    utils/fix_data_dir.sh $datadir/$part
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 50k
  # utterances in the train directory.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  #utils/subset_data_dir.sh --shortest data/SetC1_train 10000 data/SetC1_train_10kshort
  utils/subset_data_dir.sh --shortest data/SetC_train 10000 data/SetC_train_10kshort
  #randomly choose 20k utterances from the training directory
  #utils/subset_data_dir.sh data/SetC1_train 20000 data/SetC1_train_20k
  utils/subset_data_dir.sh data/SetC_train 20000 data/SetC_train_20k
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  # train a monophone system
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" \
                      data/SetC_train_10kshort data/lang_test exp/mono_C

  # decode using the monophone model
  utils/mkgraph.sh data/lang_test \
                 exp/mono_C exp/mono_C/graph_nosp_tgsmall
  for test in SetC_dev; do
    steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/mono_C/graph_nosp_tgsmall \
                  data/$test exp/mono_C/decode_nosp_tgsmall_$test
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
                    data/SetC_train_20k data/lang_test exp/mono_C exp/mono_ali_20k_C

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --cmd "$train_cmd" \
                        2000 10000 data/SetC_train_20k data/lang_test exp/mono_ali_20k_C exp/tri1_C

  # decode using the tri1 model
  utils/mkgraph.sh data/lang_test \
                  exp/tri1_C exp/tri1_C/graph_nosp_tgsmall

  for test in SetC_dev; do
    steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/tri1_C/graph_nosp_tgsmall \
                   data/$test exp/tri1_C/decode_nosp_tgsmall_$test
    
    #steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
    #               data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
    
    #steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
    #               data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
  done
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
                    data/SetC_train data/lang_test exp/tri1_C exp/tri1_ali_C


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/SetC_train data/lang_test exp/tri1_ali_C exp/tri2b_C

  # decode using the LDA+MLLT model
  
  utils/mkgraph.sh data/lang_test \
                    exp/tri2b_C exp/tri2b_C/graph_nosp_tgsmall
  for test in SetC_dev; do
    steps/decode.sh --nj 2 --cmd "$decode_cmd" exp/tri2b_C/graph_nosp_tgsmall \
                    data/$test exp/tri2b_C/decode_nosp_tgsmall_$test
   
    #steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
    #                data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
    
    #steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
    #                data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
  done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 8 --cmd "$train_cmd" --use-graphs true \
                    data/SetC_train data/lang_test exp/tri2b_C exp/tri2b_ali_C

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                    data/SetC_train data/lang_test exp/tri2b_ali_C exp/tri3b_C

  # decode using the tri3b model
  utils/mkgraph.sh data/lang_test \
                    exp/tri3b_C exp/tri3b_C/graph_nosp_tgsmall

  for test in SetC_dev; do
    steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" \
                          exp/tri3b_C/graph_nosp_tgsmall data/$test \
                          exp/tri3b_C/decode_nosp_tgsmall_$test
    #steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
    #                data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
    #steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
    #                data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
  done
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
echo ============================================================================
echo "Stage 10 Align based on tri3b, with train_all, Output tri3b_ali, trained using SAT, Output tri6b"
echo ============================================================================
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/SetC_train data/lang_test exp/tri3b_C exp/tri3b_ali_C

  #4200 40000
  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 3000 23000 \
                      data/SetC_train data/lang_test \
                      exp/tri3b_ali_C exp/tri6b_C

  # decode using the tri4b model
  utils/mkgraph.sh data/lang_test \
                     exp/tri6b_C exp/tri6b_C/graph_nosp_tgsmall

  for test in  SetC_dev; do
    steps/decode_fmllr.sh --nj 2 --cmd "$decode_cmd" \
                          exp/tri6b_C/graph_nosp_tgsmall data/$test \
                          exp/tri6b_C/decode_nosp_tgsmall_$test
  done
fi
<<!
if [ $stage -le 11 ]; then
  
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri3b exp/tri3b_ali

  #4200 40000
  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 3000 23000 \
                      data/train data/lang_nosp \
                      exp/tri3b_ali exp/tri6b

  # decode using the tri4b model
  utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri6b exp/tri6b/graph_nosp_tgsmall

  for test in test_cid_5_6 test_cid_5_15 dev test jibo_single; do
    steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
                          exp/tri6b/graph_nosp_tgsmall data/$test \
                          exp/tri6b/decode_nosp_tgsmall_$test
    
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
                         data/$test exp/tri6b/decode_nosp_{tgsmall,tgmed}_$test
    
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$test exp/tri6b/decode_nosp_{tgsmall,tglarge}_$test
    
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
      data/$test exp/tri6b/decode_nosp_{tgsmall,fglarge}_$test
  done

fi

<<!
for test in test_clean test_other dev_clean dev_other; do
    steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
                          exp/tri6b/graph_nosp_tgsmall ../../librispeech/sspapl/data/$test \
                          exp/tri6b/decode_nosp_tgsmall_libri_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
                       ../../librispeech/sspapl/data/$test exp/tri6b/decode_nosp_{tgsmall,tgmed}_libri_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      ../../librispeech/sspapl/data/$test exp/tri6b/decode_nosp_{tgsmall,tglarge}_libri_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,fglarge} \
      ../../librispeech/sspapl/data/$test exp/tri6b/decode_nosp_{tgsmall,fglarge}_libri_$test
done

if [ $stage -le 12 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  
  #steps/get_prons.sh --cmd "$train_cmd" \
  #                   data/train data/lang_nosp exp/tri4b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_nosp \
                                  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
                                  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
fi

if [ $stage -le 13 ]; then
  #train another sat system with data/lang
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train data/lang exp/tri4b exp/tri5b

  # decode using the tri4b model
  utils/mkgraph.sh data/lang_test_tgsmall \
                     exp/tri5b exp/tri5b/graph_tgsmall

  for test in test_cid_5_6 test_cid_5_15 dev test jibo_single; do
    steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
                          exp/tri5b/graph_tgsmall data/$test \
                          exp/tri5b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                         data/$test exp/tri5b/decode_{tgsmall,tgmed}_$test
    
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri5b/decode_{tgsmall,tglarge}_$test
    
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
      data/$test exp/tri5b/decode_{tgsmall,fglarge}_$test
  done
fi
!
