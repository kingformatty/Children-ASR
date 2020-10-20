
# SPAPL
# scripts to run dnn with pykaldi2

stage=3
stop_stage=3
datadir=data
expdir=exp
gmmdir=tri4b_SetC_train
pretrain_model=best_model.tar
. ./cmd.sh
. ./path.sh
. parse_options.sh

set -e

train=SetC_train
dev=SetC_dev
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # align train data using the tri4b model
  steps/align_fmllr.sh --nj 2 --cmd "$decode_cmd" \
    $datadir/$train $datadir/lang $expdir/$gmmdir $expdir/${gmmdir}_ali_$train

  # Data preparation for pykaldi2d
  local/make_pk2_data.sh --feat_type 'wav' $datadir/$train $expdir/${gmmdir}_ali_$train \
    $datadir/$train/pdfid_$gmmdir $datadir/$train/transid_$gmmdir

  # align for develope set
  steps/align_fmllr.sh --nj 2 --cmd "$decode_cmd" \
    $datadir/$dev $datadir/lang $expdir/$gmmdir $expdir/${gmmdir}_ali_$dev

  # Convert to txt
  local/make_pk2_data.sh --feat_type 'wav' $datadir/$dev $expdir/${gmmdir}_ali_$dev \
    $datadir/$dev/pdfid_$gmmdir $datadir/$dev/transid_$gmmdir
fi

if [ $stage -le 2 ] && [ $stage -ge 2 ]; then
  exp_dir=$expdir/seq_utsf_apc_l2_all_blstm_4x512_dp02_lr2e_4_l3r3_C_3foldfm/
  [ ! -d $exp_dir ] && mkdir -p $exp_dir;

  CUDA_VISIBLE_DEVICES='0' train_ce.py -train_config configs/ce.yaml \
    -dataconfig configs/data.yaml \
    -exp_dir $exp_dir\
    -lr 0.0002 \
    -net_type "lstm" \
    -feat_type "wav" \
    -batch_size 8 \
    -sweep_size 49.3 \
    -anneal_lr_epoch 5 \
    -num_epochs 10 \
    -anneal_lr_ratio 0.5 \
    -print_freq 200 \
    -resume_from_model $pretrain_model > $exp_dir/train.log 2>&1 &
    
    #
fi

  #SetC2_dev"
if [ $stage -le 3 ] && [ $stage -ge 3 ]; then
  #decoding
  exp_dir=$expdir/seq_utsf_apc_l2_all_blstm_4x512_dp02_lr2e_4_l3r3_C_3foldfm/
  
  for test in SetC1_dev SetC2_dev ; do
    local/decode_pk2.sh --nj 2 --recreate false --stage 0 --net_type 'lstm' --feat_type 'wav' --batch_size 4 \
      $datadir/$test $expdir/$gmmdir $expdir/$gmmdir/graph $exp_dir $exp_dir/decode_${test}
  done
fi


